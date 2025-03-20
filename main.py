import json
import uuid
import requests
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum


# 태스크 상태 열거형
class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


# 태스크 타입 열거형
class TaskType(str, Enum):
    EXECUTION = "execution"  # 직접 실행 가능한 태스크
    DECOMPOSITION = "decomposition"  # 분해가 필요한 태스크


# 태스크 클래스 정의
@dataclass
class Task:
    task_id: str
    description: str
    status: TaskStatus
    type: Optional[TaskType] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = None
    result: Any = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict] = None

    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []


# 로컬 LLM 클라이언트
class LLMClient:
    def __init__(self, api_url: str, model: str):
        self.api_url = api_url
        self.model = model

    def query(self, prompt: str) -> str:
        """LLM에 쿼리를 보내고 응답을 받는 메서드"""
        # Ollama API 호출
        try:
            response = requests.post(
                f"{self.api_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            print(f"LLM 쿼리 오류: {e}")
            return ""


# 도구 레지스트리
class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register_tool(self, name: str, func):
        """도구를 레지스트리에 등록"""
        self.tools[name] = func

    def execute_tool(self, name: str, args: Dict) -> Any:
        """등록된 도구 실행"""
        if name not in self.tools:
            raise ValueError(f"도구 '{name}'이(가) 등록되지 않았습니다.")
        return self.tools[name](**args)


# 태스크 매니저
class TaskManager:
    def __init__(self):
        self.tasks = {}  # task_id -> Task
        self.root_tasks = []  # 최상위 태스크 ID 목록

    def create_task(self, description: str, parent_id: Optional[str] = None) -> str:
        """새 태스크 생성"""
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            description=description,
            status=TaskStatus.PENDING,
            parent_id=parent_id,
        )
        self.tasks[task_id] = task

        if parent_id:
            parent_task = self.tasks.get(parent_id)
            if parent_task:
                parent_task.children_ids.append(task_id)
        else:
            self.root_tasks.append(task_id)

        return task_id

    def get_task(self, task_id: str) -> Optional[Task]:
        """태스크 ID로 태스크 조회"""
        return self.tasks.get(task_id)

    def update_task(self, task_id: str, **kwargs) -> bool:
        """태스크 업데이트"""
        task = self.tasks.get(task_id)
        if not task:
            return False

        for key, value in kwargs.items():
            if hasattr(task, key):
                setattr(task, key, value)

        return True

    def get_next_pending_task(self) -> Optional[Task]:
        """실행 가능한 다음 태스크 가져오기 (의존성을 고려)"""
        for task_id, task in self.tasks.items():
            if task.status == TaskStatus.PENDING:
                # 자식 태스크가 있으면 모두 완료되었는지 확인
                if task.children_ids:
                    children_completed = all(
                        self.tasks[child_id].status == TaskStatus.COMPLETED
                        for child_id in task.children_ids
                    )
                    if not children_completed:
                        continue
                return task
        return None

    def all_tasks_completed(self) -> bool:
        """모든 태스크가 완료되었는지 확인"""
        return all(
            task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            for task in self.tasks.values()
        )

    def get_task_tree(self) -> Dict:
        """태스크 트리 구조 반환"""

        def build_tree(task_id):
            task = self.tasks[task_id]
            tree = asdict(task)

            if task.children_ids:
                tree["children"] = [
                    build_tree(child_id) for child_id in task.children_ids
                ]

            return tree

        return [build_tree(root_id) for root_id in self.root_tasks]


# 자율 에이전트
class AutonomousAgent:
    def __init__(self, llm_client: LLMClient, tool_registry: ToolRegistry):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.task_manager = TaskManager()

    def _create_task_analysis_prompt(self, task: Task) -> str:
        """태스크 분석을 위한 프롬프트 생성"""
        # 사용 가능한 도구 목록
        available_tools = list(self.tool_registry.tools.keys())

        prompt = f"""
당신은 태스크를 분석하여 실행 계획을 세우는 에이전트입니다.
다음 태스크를 분석하고, 두 가지 중 하나를 선택하세요:
1. 태스크를 더 작은 서브태스크로 분할
2. 사용 가능한 도구를 사용하여 태스크를 직접 실행

태스크: {task.description}

사용 가능한 도구: {', '.join(available_tools)}

응답은 다음 JSON 형식으로 제공하세요:
```
{{
    "task_type": "decomposition 또는 execution",
    "reasoning": "당신의 결정에 대한 이유",
    "subtasks": ["서브태스크 1", "서브태스크 2", ...] // task_type이 decomposition인 경우에만
    "tool": "도구명", // task_type이 execution인 경우에만
    "tool_args": {{}} // task_type이 execution인 경우에만
}}
```

정확한 JSON 형식으로 응답해주세요.
"""
        return prompt

    def analyze_task(self, task: Task) -> Dict:
        """LLM을 사용하여 태스크 분석"""
        prompt = self._create_task_analysis_prompt(task)
        response = self.llm_client.query(prompt)

        # JSON 응답 추출 및 파싱
        try:
            # JSON 응답이 ```로 감싸져 있을 경우 처리
            if "```json" in response or "```" in response:
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]

            analysis = json.loads(response.strip())
            return analysis
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}")
            print(f"원본 응답: {response}")
            # 기본값 반환
            return {
                "task_type": "execution",
                "reasoning": "응답을 파싱할 수 없음",
                "tool": "echo",
                "tool_args": {"message": "태스크 분석 실패"},
            }

    def execute_task(self, task: Task) -> Any:
        """도구를 사용하여 태스크 실행"""
        if not task.tool_name or not task.tool_args:
            return {"error": "도구 정보가 없습니다."}

        try:
            result = self.tool_registry.execute_tool(task.tool_name, task.tool_args)
            return result
        except Exception as e:
            return {"error": str(e)}

    def run(self, initial_task: str):
        """에이전트 실행"""
        # 초기 태스크 생성
        root_task_id = self.task_manager.create_task(initial_task)

        # 태스크가 남아있는 동안 반복
        while not self.task_manager.all_tasks_completed():
            next_task = self.task_manager.get_next_pending_task()
            if not next_task:
                break

            # 태스크 진행 중으로 상태 변경
            self.task_manager.update_task(
                next_task.task_id, status=TaskStatus.IN_PROGRESS
            )

            # 태스크 분석
            analysis = self.analyze_task(next_task)
            task_type = analysis.get("task_type")

            if task_type == "decomposition":
                # 태스크를 서브태스크로 분해
                self.task_manager.update_task(
                    next_task.task_id, type=TaskType.DECOMPOSITION
                )

                # 서브태스크 생성
                subtasks = analysis.get("subtasks", [])
                for subtask in subtasks:
                    self.task_manager.create_task(subtask, parent_id=next_task.task_id)

                # 현재 태스크 완료로 표시
                self.task_manager.update_task(
                    next_task.task_id, status=TaskStatus.COMPLETED
                )
                print(
                    f"태스크 분해: {next_task.description} -> {len(subtasks)}개의 서브태스크"
                )

            elif task_type == "execution":
                # 태스크 실행
                self.task_manager.update_task(
                    next_task.task_id,
                    type=TaskType.EXECUTION,
                    tool_name=analysis.get("tool"),
                    tool_args=analysis.get("tool_args", {}),
                )

                # 도구 실행
                result = self.execute_task(next_task)

                # 결과 저장 및 태스크 완료로 표시
                self.task_manager.update_task(
                    next_task.task_id,
                    result=result,
                    status=(
                        TaskStatus.COMPLETED
                        if "error" not in result
                        else TaskStatus.FAILED
                    ),
                )

                print(f"태스크 실행: {next_task.description} -> {result}")

        # 최종 결과 반환
        return self.task_manager.get_task_tree()


# 샘플 도구 함수들
def echo(message: str) -> Dict:
    return {"message": message}


def add(a: int, b: int) -> Dict:
    return {"result": a + b}


def fetch_web_content(url: str) -> Dict:
    try:
        response = requests.get(url)
        response.raise_for_status()
        return {"status": "success", "content": response.text[:500] + "..."}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# 메인 실행 코드
def main():
    # LLM 클라이언트 초기화 (Ollama 사용)
    llm_client = LLMClient(api_url="http://localhost:11434", model="llama3")

    # 도구 레지스트리 초기화 및 도구 등록
    tool_registry = ToolRegistry()
    tool_registry.register_tool("echo", echo)
    tool_registry.register_tool("add", add)
    tool_registry.register_tool("fetch_web_content", fetch_web_content)

    # 에이전트 초기화
    agent = AutonomousAgent(llm_client, tool_registry)

    # 초기 태스크
    initial_task = "웹에서 최신 파이썬 라이브러리 정보를 찾아서 요약해줘"

    # 에이전트 실행
    result = agent.run(initial_task)

    # 결과 출력
    print("\n=== 최종 태스크 트리 ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
