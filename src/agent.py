import logging
import asyncio
from dataclasses import dataclass, asdict, is_dataclass

import requests
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AgentTask,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    RunContext,
    ToolError,
    cli,
    function_tool,
    get_job_context,
    inference,
    llm,
    room_io,
    utils,
)
from livekit.agents.beta.tools import EndCallTool
from livekit.agents.beta.workflows import TaskGroup
from livekit.agents.llm.chat_context import FunctionCall
from livekit.agents.llm.utils import execute_function_call
from livekit.plugins import (
    ai_coustics,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("my-agent")

load_dotenv(".env.local")

def send_to_wecom(data):
    webhook="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=8dddf9e3-36d8-4fe5-8c4d-a9f243389e1e"

    if not webhook:
        logger.error("未配置企业微信 webhook")
        return

    try:
        reason = data.get("reason") or "无"

        text = f"""
来访登记信息

姓名：{data.get('name', '')}
电话：{data.get('number', '')}
车牌号：{data.get('plate', '')}
到访单位：{data.get('company', '')}
到访事由：{reason}
"""

        payload = {
            "msgtype": "text",
            "text": {
                "content": text.strip()
            }
        }

        r = requests.post(
            webhook,
            json=payload,
            timeout=10
        )

        logger.info(f"企业微信发送成功: {r.text}")

    except Exception as e:
        logger.exception(f"发送企业微信失败: {e}")


def _to_json_serializable(obj):
    """Convert dataclasses and nested structures to JSON-serializable form."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, list):
        return [_to_json_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    return obj

@dataclass
class ContactIdentificationResults:
    name: str
    plate: str
    number: str
    company: str
    reason: str | None = None

class ContactIdentificationTask(AgentTask):
    def __init__(self, agent_instructions: str, extra_tools: list | None = None):
        no_greet_prefix = "直接询问你需要的信息.\n"
        task_instructions = "- 收集姓名，电话号，车牌号，到访单位和到访事由."
        no_goodbye_suffix = "\n重要: 不要说再见或你已完成，只需要快速向用户再确认一遍收集好的信息并马上用工具记录下来"
        wrapped_instructions = no_greet_prefix + agent_instructions + "\n" + task_instructions + no_goodbye_suffix
        super().__init__(
            instructions=wrapped_instructions,
            tools=list(extra_tools) if extra_tools else [],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions=(
                "Begin this task now. If the task instructions require calling "
                "a tool first (for example, to look up information), call it. "
                "Otherwise, ask the user for the information described in your "
                "task instructions."
            ),
            allow_interruptions=True,
            tool_choice="auto",
        )

    @function_tool(name="record_contact_identification")
    async def record_contact_identification(
        self,
        context: RunContext,
            name: str,
            plate: str,
            number: str,
            company: str,
            reason: str | None = None
    ):
        """Call when you have collected all required data points for this task.
Provide the structured results exactly as requested.
Do not confirm on record, remain silent and move to the next task.

Args:
    name: str,
    plate: str,
    number: str,
    company: str,
    reason: str """
        self.complete(ContactIdentificationResults(name=name,plate=plate,number=number,company=company,reason=reason))

class DefaultAgent(Agent):
    def __init__(self) -> None:
        self._agent_instructions = """你是一个中文门卫接待。
你必须遵守以下规则：
1. 全程只使用简体中文回答
2. 不允许使用英文单词（除非是专有名词）
3. 输出必须自然口语化
4. 简洁，不要列表，不要符号，不要表情"""
        super().__init__(
            instructions="",
        )
    async def on_enter(self):
        greeting_instructions = ""
        greeting_instructions = """让来访者把姓名，电话，车牌号，到访单位和到访事由说一下"""
        # The greeting must not ask a question — the first data collection task
        # asks the opening question. Without this guardrail the LLM tends to end
        # with an open-ended prompt ("How can I help?"), which collides with the
        # task's first turn.
        no_question_guardrail = (
            "IMPORTANT: The greeting must be a statement only. Do NOT end with any "
            'question, including open-ended prompts like "How can I help?". The '
            "next task will ask the first question."
        )
        await self.session.generate_reply(
            instructions="\n".join(
                part for part in (self._agent_instructions, greeting_instructions, no_question_guardrail) if part
            ),
            allow_interruptions=True,
        )
        # Propagate HTTP/client/MCP tools into each data collection task so
        # they're callable mid-task (e.g. looking up a customer record while
        # collecting details). EndCallTool is excluded here — it's invoked
        # programmatically in _finish_data_collection.
        _task_tools = [t for t in self.tools if not isinstance(t, EndCallTool)]
        task_group = TaskGroup(chat_ctx=self.chat_ctx)
        task_group.add(
            lambda _ai=self._agent_instructions, _tools=_task_tools: ContactIdentificationTask(agent_instructions=_ai, extra_tools=_tools),
            id="contact_identification",
            description="收集来访者的姓名，电话，车牌号，到访单位和到访事由",
        )

        try:
            group_result = await task_group
        except (ToolError, asyncio.CancelledError):
            logger.info("data collection task group cancelled (participant likely disconnected)")
            return

        await self._finish_data_collection(group_result.task_results)
    async def _finish_data_collection(self, task_results):
        """Serialize results, speak goodbye, and end the session."""
        serialized = _to_json_serializable(task_results)
        get_job_context().proc.userdata["dc_results"] = serialized
        try:

            task = serialized.get(
                "contact_identification",
                {}
            )

            if isinstance(task, dict) and "result" in task:
                result = task["result"]
            else:
                result = task

            await asyncio.to_thread(
                send_to_wecom,
                result
            )

        except Exception as e:

            logger.exception(
                f"处理登记结果失败: {e}"
            )

        end_instructions = """好的，已通知门卫，再见."""

        summary_task: asyncio.Task | None = None

        # Remove EndCallTool from active tools so the LLM cannot call it
        # spontaneously during the goodbye speech (it is invoked programmatically below).
        await self.update_tools([t for t in self.tools if not isinstance(t, EndCallTool)])

        speech_handle = self.session.generate_reply(
            instructions=f"信息收集完成 {end_instructions}",
            tool_choice="none",
        )

        try:
            await speech_handle
            if summary_task:
                await summary_task
        except ConnectionError:
            logger.debug("user disconnected during goodbye speech")

        try:
            end_call_tool = next((t for t in self.tools if isinstance(t, EndCallTool)), None)
            if not end_call_tool:
                end_call_tool = EndCallTool(
                    end_instructions=end_instructions,
                    delete_room=False,
                )

            tools_with_end_call = [*self.tools, end_call_tool]
            tool_ctx = llm.ToolContext(tools_with_end_call)
            end_call_id = utils.shortuuid("fnc_")
            tool_call = llm.FunctionToolCall(
                call_id=end_call_id,
                name="end_call",
                arguments="{}",
            )
            fnc_call = FunctionCall(
                call_id=end_call_id,
                name="end_call",
                arguments="{}",
            )
            call_ctx = RunContext(
                session=self.session,
                speech_handle=speech_handle,
                function_call=fnc_call,
            )
            await execute_function_call(
                tool_call,
                tool_ctx,
                call_ctx=call_ctx,
            )
        except (ConnectionError, RuntimeError):
            logger.debug("room already disconnected during end-call teardown")


server = AgentServer()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

server.setup_fnc = prewarm

@server.rtc_session(agent_name="my-agent")
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=inference.STT(model="elevenlabs/scribe_v2_realtime", language="zh"),
        llm=inference.LLM(
            model="openai/gpt-5.2-chat-latest",
            extra_kwargs={"reasoning_effort": "low"},
        ),
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="653b9445-ae0c-4312-a3ce-375504cff31e",
            language="zh"
        ),
        turn_handling=TurnHandlingOptions(turn_detection=MultilingualModel()),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )
    ctx.proc.userdata["dc_results"] = None

    await session.start(
        agent=DefaultAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_L,
                ),
            ),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
