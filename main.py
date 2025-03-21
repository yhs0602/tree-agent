import json
import re
import uuid
import requests
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
import hashlib


# 태스크 상태 열거형 (JSON 직렬화 가능)
class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


# 태스크 타입 열거형 (JSON 직렬화 가능)
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
    children_ids: List[str] = field(default_factory=list)
    result: Any = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict] = None
    failure_reason: Optional[str] = None  # 실패 이유 추가
    retry_count: int = 0  # 재시도 횟수 추가

    # 태스크 해시 생성 (중복 감지용)
    @property
    def hash(self) -> str:
        return hashlib.md5(self.description.lower().strip().encode()).hexdigest()


# JSON 직렬화 가능한 Enum을 위한 인코더
class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


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
        self.task_hashes = {}  # task_hash -> task_id (중복 감지용)
        self.root_tasks = []  # 최상위 태스크 ID 목록

    def create_task(self, description: str, parent_id: Optional[str] = None) -> str:
        """새 태스크 생성 (중복 감지 및 재사용)"""
        # 태스크 설명에서 해시 생성
        task_hash = hashlib.md5(description.lower().strip().encode()).hexdigest()

        # 이미 존재하는 태스크인지 확인
        if task_hash in self.task_hashes:
            existing_task_id = self.task_hashes[task_hash]

            # 부모 태스크가 있는 경우 자식으로 연결
            if parent_id:
                parent_task = self.tasks.get(parent_id)
                if parent_task and existing_task_id not in parent_task.children_ids:
                    parent_task.children_ids.append(existing_task_id)

            return existing_task_id

        # 새 태스크 생성
        task_id = str(uuid.uuid4())
        task = Task(task_id=task_id, description=description, status=TaskStatus.PENDING)
        self.tasks[task_id] = task
        self.task_hashes[task_hash] = task_id

        # 부모-자식 관계 설정
        if parent_id:
            task.parent_id = parent_id
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

    def get_similar_tasks(self, description: str, limit: int = 5) -> List[Task]:
        """유사한 태스크 찾기 (단순 키워드 매칭)"""
        keywords = set(description.lower().split())
        task_scores = []

        for task_id, task in self.tasks.items():
            if task.status == TaskStatus.COMPLETED:
                task_keywords = set(task.description.lower().split())
                common_words = keywords.intersection(task_keywords)
                score = len(common_words) / max(len(keywords), len(task_keywords))

                if score > 0.2:  # 일정 유사도 이상인 경우만
                    task_scores.append((score, task))

        # 유사도 기준으로 정렬하여 상위 N개 반환
        similar_tasks = [
            task
            for _, task in sorted(task_scores, key=lambda x: x[0], reverse=True)[:limit]
        ]
        return similar_tasks


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

        # 유사한 태스크 목록 (재사용 가능한 서브태스크 제안)
        similar_tasks = self.task_manager.get_similar_tasks(task.description)

        # 유사한 태스크의 분해 결과 추출
        reusable_subtasks = []
        for similar_task in similar_tasks:
            if similar_task.type == TaskType.DECOMPOSITION:
                child_tasks = [
                    self.task_manager.get_task(child_id).description
                    for child_id in similar_task.children_ids
                ]
                reusable_subtasks.extend(child_tasks)

        # 중복 제거하고 최대 10개 유지
        reusable_subtasks = list(set(reusable_subtasks))[:10]

        # 이미 생성된 서브태스크 목록을 프롬프트에 추가
        reusable_subtasks_text = "\n".join(
            [f"- {subtask}" for subtask in reusable_subtasks]
        )

        # 이전 실패 정보 추가
        failure_info = ""
        if task.failure_reason and task.retry_count > 0:
            failure_info = f"""
This task has failed {task.retry_count} times before.
Failure reason: {task.failure_reason}

Consider trying a different approach based on the previous failures.
"""

        prompt = f"""
You are an agent responsible for analyzing tasks and creating execution plans.
Analyze the given task and choose one of the following two options:
1. Decompose the task into smaller subtasks.
2. Use an available tool to execute the task directly.

Task: {task.description}

Available tools: {', '.join(available_tools)}
{failure_info}

Previously generated subtasks (available for reuse):
{reusable_subtasks_text if reusable_subtasks else ""}

Reuse guidelines:
- Minimize duplication by reusing existing subtasks whenever possible.
- If a new subtask is similar to an existing one, use the exact same description.
- When creating new subtasks, write them in a general format to allow for future reuse.

Provide your response in the following JSON format:

```
{{
    "task_type": "decomposition or execution",
    "reasoning": "Your reasoning for the decision",
    "subtasks": ["Subtask 1", "Subtask 2", ...] // Only for task_type = decomposition
    "tool": "Tool name", // Only for task_type = execution
    "tool_args": {{}} // Only for task_type = execution
}}
```

Ensure the response is in a valid JSON format.
Do not use comments, and `subtasks` must be a list of strings.
Do not use placeholders—provide actual values. Whenever possible, use the given tools to complete the task.
"""
        return prompt

    def analyze_task(self, task: Task) -> Tuple[Dict, bool]:
        """LLM을 사용하여 태스크 분석"""
        prompt = self._create_task_analysis_prompt(task)
        response = self.llm_client.query(prompt)

        # JSON 응답 추출 및 파싱
        try:
            # Think token 제거
            response = re.sub(
                r"<think>.*?</think>", "", response, flags=re.DOTALL
            ).strip()
            # JSON 응답이 ```로 감싸져 있을 경우 처리
            if "```json" in response or "```" in response:
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            else:
                # 코드 블록이 없고 전체가 JSON이라고 가정
                response = response.strip()

            analysis = json.loads(response.strip())

            # 응답 유효성 검증
            if analysis.get("task_type") == "decomposition":
                if not analysis.get("subtasks") or not isinstance(
                    analysis["subtasks"], list
                ):
                    return {
                        "task_type": "execution",
                        "reasoning": "서브태스크가 없거나 유효하지 않음",
                        "tool": "echo",
                        "tool_args": {
                            "message": "태스크 분석 실패: 유효하지 않은 서브태스크"
                        },
                        "error": "태스크 분석 실패: 서브태스크가 없거나 유효하지 않음",
                    }, False

                if not all(
                    isinstance(subtask, str) for subtask in analysis["subtasks"]
                ):
                    wrong_subtasks = [
                        subtask
                        for subtask in analysis["subtasks"]
                        if not isinstance(subtask, str)
                    ]
                    return {
                        "task_type": "execution",
                        "reasoning": f"서브태스크는 문자열 목록이어야 함: 위반: {wrong_subtasks}",
                        "tool": "echo",
                        "tool_args": {
                            "message": f"서브태스크는 문자열 목록이어야 함: 위반: {wrong_subtasks}"
                        },
                        "error": f"태스크 분석 실패: 서브태스크는 문자열 목록이어야 함; 위반: {wrong_subtasks}",
                    }, False

            elif analysis.get("task_type") == "execution":
                if not analysis.get("tool"):
                    return {
                        "task_type": "execution",
                        "reasoning": "도구 이름이 지정되지 않음",
                        "tool": "echo",
                        "tool_args": {"message": "태스크 분석 실패: 도구 이름 누락"},
                        "error": "태스크 분석 실패: 도구 이름이 지정되지 않음",
                    }, False

                if analysis.get("tool") not in self.tool_registry.tools:
                    return {
                        "task_type": "execution",
                        "reasoning": f"도구 '{analysis.get('tool')}'이(가) 등록되지 않음",
                        "tool": "echo",
                        "tool_args": {
                            "message": f"태스크 분석 실패: 없는 도구 '{analysis.get('tool')}'"
                        },
                        "error": f"태스크 분석 실패: 도구 '{analysis.get('tool')}'이(가) 등록되지 않음",
                    }, False

            else:
                return {
                    "task_type": "execution",
                    "reasoning": "유효하지 않은 task_type",
                    "tool": "echo",
                    "tool_args": {
                        "message": "태스크 분석 실패: task_type은 'decomposition' 또는 'execution'이어야 함"
                    },
                    "error": "태스크 분석 실패: task_type은 'decomposition' 또는 'execution'이어야 함",
                }, False

            return analysis, True

        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}")
            print(f"원본 응답: {response}")
            # 기본값 반환
            return {
                "task_type": "execution",
                "reasoning": f"JSON 파싱 오류: {e}",
                "tool": "echo",
                "tool_args": {"message": f"태스크 분석 실패: JSON 파싱 오류 - {e}"},
                "error": f"태스크 분석 실패: JSON 파싱 오류 - {e}",
            }, False

    def execute_task(self, task: Task) -> Any:
        """도구를 사용하여 태스크 실행"""
        if not task.tool_name or not task.tool_args:
            return {"error": "도구 정보가 없습니다."}

        try:
            result = self.tool_registry.execute_tool(task.tool_name, task.tool_args)
            return result
        except Exception as e:
            return {"error": str(e)}

    def run(self, initial_task: str, max_retries: int = 3):
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
            analysis, success = self.analyze_task(next_task)

            if not success:
                # 분석 실패 처리
                failure_reason = f"태스크 분석 실패: {analysis.get('error')}"
                retry_count = next_task.retry_count + 1

                if retry_count <= max_retries:
                    self.task_manager.update_task(
                        next_task.task_id,
                        status=TaskStatus.PENDING,
                        failure_reason=failure_reason,
                        retry_count=retry_count,
                    )
                    print(
                        f"태스크 분석 실패: {next_task.description}. 재시도 {retry_count}/{max_retries} ({failure_reason})"
                    )
                else:
                    self.task_manager.update_task(
                        next_task.task_id,
                        status=TaskStatus.FAILED,
                        failure_reason=failure_reason,
                    )
                    print(
                        f"태스크 분석 최종 실패: {next_task.description}. 최대 재시도 횟수 초과"
                    )
                continue

            task_type = analysis.get("task_type")

            if task_type == "decomposition":
                # 태스크를 서브태스크로 분해
                self.task_manager.update_task(
                    next_task.task_id, type=TaskType.DECOMPOSITION
                )

                # 서브태스크 생성 (중복 감지 및 재사용)
                subtasks = analysis.get("subtasks", [])
                for subtask in subtasks:
                    self.task_manager.create_task(subtask, parent_id=next_task.task_id)

                # 현재 태스크 완료로 표시
                self.task_manager.update_task(
                    next_task.task_id, status=TaskStatus.COMPLETED
                )
                print(
                    f"태스크 분해: {next_task.description} -> {len(subtasks)}개의 서브태스크: {subtasks}"
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
                if "error" in result:
                    # 실패 처리
                    failure_reason = f"도구 실행 오류: {result['error']}"
                    retry_count = next_task.retry_count + 1

                    if retry_count <= max_retries:
                        # 재시도를 위해 상태를 PENDING으로 설정하고 실패 이유 기록
                        self.task_manager.update_task(
                            next_task.task_id,
                            status=TaskStatus.PENDING,
                            failure_reason=failure_reason,
                            retry_count=retry_count,
                        )
                        print(
                            f"태스크 실패: {next_task.description}. 재시도 {retry_count}/{max_retries}. 이유: {failure_reason}"
                        )
                    else:
                        # 최대 재시도 횟수 초과
                        self.task_manager.update_task(
                            next_task.task_id,
                            result=result,
                            status=TaskStatus.FAILED,
                            failure_reason=failure_reason,
                        )
                        print(
                            f"태스크 최종 실패: {next_task.description}. 최대 재시도 횟수 초과. 이유: {failure_reason}"
                        )
                else:
                    # 성공 처리
                    self.task_manager.update_task(
                        next_task.task_id, result=result, status=TaskStatus.COMPLETED
                    )
                    print(f"태스크 실행 성공: {next_task.description}")

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


def parse_html(html_content: str, selector: str = None) -> Dict:
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_content, "html.parser")

        if selector:
            elements = soup.select(selector)
            return {
                "status": "success",
                "elements": [str(el) for el in elements[:5]],
                "count": len(elements),
            }
        else:
            return {
                "status": "success",
                "title": soup.title.string if soup.title else "No title",
                "text": soup.get_text()[:500] + "...",
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# 메인 실행 코드
def main():
    # LLM 클라이언트 초기화 (Ollama 사용)
    llm_client = LLMClient(api_url="http://localhost:11434", model="qwq4-4bit:latest")

    # 도구 레지스트리 초기화 및 도구 등록
    tool_registry = ToolRegistry()
    tool_registry.register_tool("echo", echo)
    tool_registry.register_tool("add", add)
    tool_registry.register_tool("fetch_web_content", fetch_web_content)
    tool_registry.register_tool("parse_html", parse_html)

    # 에이전트 초기화
    agent = AutonomousAgent(llm_client, tool_registry)

    # 초기 태스크
    initial_task = "파이썬 라이브러리 정보를 찾기 위해 Python.org와 PyPi.org를 조사하고 결과를 요약해줘"

    # 에이전트 실행
    result = agent.run(initial_task)

    # 결과 출력 (EnumEncoder 사용)
    print("\n=== 최종 태스크 트리 ===")
    print(json.dumps(result, indent=2, ensure_ascii=False, cls=EnumEncoder))


if __name__ == "__main__":
    main()
