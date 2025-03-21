import json
import re
import uuid
import requests
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import hashlib
from collections import deque


# Element types in the stack machine
class ElementType(str, Enum):
    TASK = "task"  # A task to be executed
    OPERAND = "operand"  # Data/result from a task execution
    OPERATOR = "operator"  # Operation to be performed on operands
    CONTROL = "control"  # Control flow instruction


# Task status enum (JSON serializable)
class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


# Task type enum (JSON serializable)
class TaskType(str, Enum):
    EXECUTION = "execution"  # Directly executable task
    DECOMPOSITION = "decomposition"  # Task that needs decomposition
    RESPONSE = "response"  # Task that requires a response


# Stack Element Base Class
@dataclass
class StackElement:
    element_id: str
    element_type: ElementType


# Task class definition (as a type of stack element)
@dataclass
class Task(StackElement):
    description: str
    status: TaskStatus
    type: Optional[TaskType] = None
    result: Any = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict] = None
    failure_reason: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 5
    dependencies: List[str] = field(
        default_factory=list
    )  # IDs of tasks this depends on

    # Generate task hash (for duplicate detection)
    @property
    def hash(self) -> str:
        return hashlib.md5(self.description.lower().strip().encode()).hexdigest()

    def __post_init__(self):
        if not hasattr(self, "element_type"):
            self.element_type = ElementType.TASK


# Operand class (for storing data/results)
@dataclass
class Operand(StackElement):
    value: Any
    source_id: Optional[str] = None  # ID of the task that produced this operand

    def __post_init__(self):
        if not hasattr(self, "element_type"):
            self.element_type = ElementType.OPERAND


# Operator class (for operations on operands)
@dataclass
class Operator(StackElement):
    operation: str
    args: Dict = field(default_factory=dict)
    operand_count: int = 2  # Number of operands this operator needs

    def __post_init__(self):
        if not hasattr(self, "element_type"):
            self.element_type = ElementType.OPERATOR


# Control Flow Instruction
@dataclass
class ControlInstruction(StackElement):
    instruction: str  # e.g., "LOOP", "BRANCH", "RETURN"
    args: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not hasattr(self, "element_type"):
            self.element_type = ElementType.CONTROL


# JSON Encoder for Enum serialization
class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


# Local LLM Client
class LLMClient:
    def __init__(self, api_url: str, model: str):
        self.api_url = api_url
        self.model = model

    def query(self, prompt: str) -> str:
        """Send query to LLM and get response"""
        try:
            response = requests.post(
                f"{self.api_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            print(f"LLM query error: {e}")
            return ""


# Tool Registry
class ToolRegistry:
    def __init__(self):
        self.tools = {}
        self.tools_docs = {}

    def register_tool(self, name: str, func: Callable, documentation: str):
        """Register a tool to the registry"""
        self.tools[name] = func
        self.tools_docs[name] = documentation

    def execute_tool(self, name: str, args: Dict) -> Any:
        """Execute a registered tool"""
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' is not registered.")
        try:
            return self.tools[name](**args)
        except Exception as e:
            raise ValueError(
                f"Error executing tool '{name}': {e}; "
                f"Args: {args}, Documentation: {self.tools_docs.get(name)}"
            )


# Stack Machine for the Agent (no more tree structure)
class StackMachine:
    def __init__(self):
        self.stack = deque()  # Operating stack
        self.elements = {}  # element_id -> StackElement
        self.element_hashes = {}  # element_hash -> element_id for dedup

    def push(self, element: StackElement) -> None:
        """Push an element onto the stack."""
        self.elements[element.element_id] = element
        self.stack.append(element.element_id)

        # For tasks, store a hash for duplicate detection
        if element.element_type == ElementType.TASK:
            task = element
            self.element_hashes[task.hash] = task.element_id

    def pop(self) -> Optional[StackElement]:
        """Pop and return the top element from the stack."""
        if not self.stack:
            return None
        element_id = self.stack.pop()
        return self.elements.get(element_id)

    def peek(self) -> Optional[StackElement]:
        """Look at the top element without removing it."""
        if not self.stack:
            return None
        element_id = self.stack[-1]
        return self.elements.get(element_id)

    def size(self) -> int:
        """Get current stack size."""
        return len(self.stack)

    def is_empty(self) -> bool:
        """Check if the stack is empty."""
        return len(self.stack) == 0

    def get_element(self, element_id: str) -> Optional[StackElement]:
        """Get an element by ID."""
        return self.elements.get(element_id)

    def update_element(self, element_id: str, **kwargs) -> bool:
        """Update an element by ID with provided attributes."""
        element = self.elements.get(element_id)
        if not element:
            return False

        for key, value in kwargs.items():
            if hasattr(element, key):
                setattr(element, key, value)

        return True

    def create_task(
        self,
        description: str,
        dependencies: Optional[List[str]] = None,
        max_retries: int = 3,
    ) -> str:
        """
        Create a new task (with duplicate detection).
        The new task is not tied to any parent/child structure;
        it is simply put on the stack with optional dependencies.
        """
        task_hash = hashlib.md5(description.lower().strip().encode()).hexdigest()

        # Check if this task already exists (by hash)
        if task_hash in self.element_hashes:
            existing_task_id = self.element_hashes[task_hash]
            existing_task = self.elements.get(existing_task_id)

            # Add any new dependencies (if any) to the existing task
            if dependencies:
                for dep_id in dependencies:
                    if dep_id not in existing_task.dependencies:
                        existing_task.dependencies.append(dep_id)
            return existing_task_id

        # Create new
        task_id = str(uuid.uuid4())
        task = Task(
            element_id=task_id,
            element_type=ElementType.TASK,
            description=description,
            status=TaskStatus.PENDING,
            dependencies=dependencies or [],
            max_retries=max_retries,
        )

        # Store and hash
        self.elements[task_id] = task
        self.element_hashes[task_hash] = task_id
        return task_id

    def get_next_executable_task(self) -> Optional[Task]:
        """
        Get the next executable task (PENDING + dependencies met).
        This doesn't consider a parent-child relationship at all.
        """
        for element_id in self.stack:
            element = self.elements.get(element_id)
            if not element or element.element_type != ElementType.TASK:
                continue

            task = element
            if task.status != TaskStatus.PENDING:
                continue

            # Check if all dependencies are completed
            dependencies_met = True
            for dep_id in task.dependencies:
                dep_task = self.elements.get(dep_id)
                if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                    dependencies_met = False
                    break

            if dependencies_met:
                return task

        return None

    def create_operand_from_task(self, task_id: str) -> str:
        """Create an operand from a task's result."""
        task = self.elements.get(task_id)
        if not task or task.element_type != ElementType.TASK:
            raise ValueError(f"Invalid task ID: {task_id}")

        operand_id = str(uuid.uuid4())
        operand = Operand(
            element_id=operand_id,
            element_type=ElementType.OPERAND,
            value=task.result,
            source_id=task_id,
        )
        self.elements[operand_id] = operand
        return operand_id


# Agent class using stack machine architecture
class StackMachineAgent:
    def __init__(self, llm_client: LLMClient, tool_registry: ToolRegistry):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.stack_machine = StackMachine()

    def _create_task_analysis_prompt(self, task: Task) -> str:
        """Create prompt for LLM to decide whether to execute directly or decompose."""
        available_tools = list(self.tool_registry.tools.keys())
        available_tools_with_docs = (
            f"{tool}: {self.tool_registry.tools_docs.get(tool, '')}"
            for tool in available_tools
        )

        # Add information about any completed dependency results
        dependency_info = ""
        if task.dependencies:
            completed_info = []
            for dep_id in task.dependencies:
                dep_task = self.stack_machine.get_element(dep_id)
                if dep_task and dep_task.status == TaskStatus.COMPLETED:
                    completed_info.append(
                        f"Dependency: {dep_task.description}\nResult: {dep_task.result}"
                    )
            if completed_info:
                dependency_info = (
                    "You have these completed dependencies with results:\n"
                    + "\n".join(completed_info)
                )

        # Add any previous failure info
        failure_info = ""
        if task.failure_reason and task.retry_count > 0:
            failure_info = f"""
This task has failed {task.retry_count} times before.
Failure reason: {task.failure_reason}

Consider trying a different approach based on previous failures.
"""

        prompt = f"""
You are an agent responsible for analyzing tasks and creating execution plans.
Analyze the given task and choose one of the following two options:
1. Use an available tool to execute the task directly.
2. Respond directly to complete the task.
3. Decompose the task into two (or more) subtasks.

Task: {task.description}

Available tools: {', '.join(available_tools_with_docs)}
{dependency_info}
{failure_info}

Provide your response in the following JSON format (no extra keys, no comments):

```
{{
  "task_type": "decomposition or execution or response",
  "reasoning": "Your reasoning",
  "response": "Your response",
  "subtasks": [
      {{
        "description": "Subtask 1",
        "dependencies": [] 
      }},
      {{
          "description": "Subtask 2",
          "dependencies": []
      }}
    ],
  "tool": "Tool name",
  "tool_args": {{}} 
}}
```

- When task_type = "execution", include "tool" and "tool_args" only.
- When task_type = "decomposition", include a "subtasks" array (each with a description and optional dependencies).
- When task_type = "response", provide a direct response in the "response" field.
- Do not use placeholders; provide actual values. 
- If reusing data from dependencies, place them appropriately in tool_args.
"""
        return prompt

    def analyze_task(self, task: Task) -> Tuple[Dict, bool]:
        """Analyze a task using LLM and parse the JSON response."""
        prompt = self._create_task_analysis_prompt(task)
        response = self.llm_client.query(prompt)

        # Remove any <think> ... </think> placeholders from LLM
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        # Strip code fences if present
        if "```json" in response or "```" in response:
            parts = response.split("```")
            # Attempt to pick the portion that might be JSON
            response = parts[1] if len(parts) > 1 else parts[0]

            # Sometimes there's a 'json' marker
            response = response.replace("json", "").strip()

        # Attempt to parse as JSON
        try:
            analysis = json.loads(response.strip())

            # Validate
            if analysis.get("task_type") == "decomposition":
                if not isinstance(analysis.get("subtasks"), list):
                    return {
                        "task_type": "execution",
                        "reasoning": "Invalid or missing subtasks",
                        "tool": "echo",
                        "tool_args": {"message": "Task analysis error"},
                        "error": "Task analysis failed: Subtasks missing or invalid",
                    }, False

                # Check each subtask for minimal validity
                for sub in analysis["subtasks"]:
                    if not isinstance(sub, dict) or "description" not in sub:
                        return {
                            "task_type": "execution",
                            "reasoning": "Invalid subtask format",
                            "tool": "echo",
                            "tool_args": {"message": "Task analysis error"},
                            "error": "Invalid subtask: missing description",
                        }, False

            elif analysis.get("task_type") == "execution":
                if not analysis.get("tool"):
                    return {
                        "task_type": "execution",
                        "reasoning": "Missing tool name for execution",
                        "tool": "echo",
                        "tool_args": {"message": "Task analysis error"},
                        "error": "Tool name not specified",
                    }, False

                if analysis["tool"] not in self.tool_registry.tools:
                    return {
                        "task_type": "execution",
                        "reasoning": f"Unregistered tool: {analysis['tool']}",
                        "tool": "echo",
                        "tool_args": {"message": "Task analysis error"},
                        "error": f"Tool '{analysis['tool']}' is not registered",
                    }, False
            elif analysis.get("task_type") == "response":
                if not analysis.get("response"):
                    return {
                        "task_type": "execution",
                        "reasoning": "Missing response",
                        "tool": "echo",
                        "tool_args": {"message": "Task analysis error"},
                        "error": "Response missing",
                    }, False
            else:
                return {
                    "task_type": "execution",
                    "reasoning": "Invalid or missing task_type",
                    "tool": "echo",
                    "tool_args": {"message": "Task analysis error"},
                    "error": "Task analysis failed: invalid task_type",
                }, False

            return analysis, True

        except json.JSONDecodeError as e:
            print("JSON parsing error:", e)
            return {
                "task_type": "execution",
                "reasoning": "JSON decode error",
                "tool": "echo",
                "tool_args": {
                    "message": f"Task analysis failed: JSON parsing error - {e}"
                },
                "error": f"JSON parsing error: {e}",
            }, False

    def execute_task(self, task: Task) -> Any:
        """Execute task using a specified tool."""
        if not task.tool_name or not task.tool_args:
            return {"error": "Tool information missing."}

        try:
            # If needed, you can do more sophisticated handling of dependency-based data insertion
            result = self.tool_registry.execute_tool(task.tool_name, task.tool_args)
            return result
        except Exception as e:
            return {"error": str(e)}

    def run(self, initial_task: str):
        """Run the agent in a loop until the stack is empty, with no hierarchical task tree."""
        # Create initial task and push to stack
        initial_task_id = self.stack_machine.create_task(initial_task)
        self.stack_machine.push(self.stack_machine.get_element(initial_task_id))

        while not self.stack_machine.is_empty():
            next_task = self.stack_machine.get_next_executable_task()
            if next_task is None:
                break
            print(f"Next task: {next_task.description if next_task else None}")

            if not next_task:
                # If we can't find any executable task, pop the top and see what it is
                top_element = self.stack_machine.peek()
                if not top_element:
                    break

                if top_element.element_type == ElementType.TASK:
                    task = top_element
                    # If it's completed, turn it into an operand
                    if task.status == TaskStatus.COMPLETED:
                        self.stack_machine.pop()
                        operand_id = self.stack_machine.create_operand_from_task(
                            task.element_id
                        )
                        self.stack_machine.push(
                            self.stack_machine.get_element(operand_id)
                        )
                        print(
                            f"Converted completed task to operand: {task.description}"
                        )
                    elif top_element.element_type == TaskStatus.FAILED:
                        # If it's not completed, remove it since it can't proceed
                        self.stack_machine.pop()
                        print(
                            f"Removing unexecutable task from stack: {task.description}"
                        )

                elif top_element.element_type == ElementType.OPERATOR:
                    self.stack_machine.pop()
                    operator = top_element
                    operands = []

                    for _ in range(operator.operand_count):
                        if self.stack_machine.is_empty():
                            break
                        elem = self.stack_machine.pop()
                        if elem.element_type == ElementType.OPERAND:
                            operands.append(elem)

                    if len(operands) == operator.operand_count:
                        # Suppose we do some operation (not really implemented here)
                        # Then push a new operand
                        result = {"message": f"Applied operator {operator.operation}"}
                        new_operand_id = str(uuid.uuid4())
                        new_operand = Operand(
                            element_id=new_operand_id,
                            element_type=ElementType.OPERAND,
                            value=result,
                        )
                        self.stack_machine.elements[new_operand_id] = new_operand
                        self.stack_machine.push(new_operand)
                    else:
                        # Return the partial operands to the stack
                        for operand in reversed(operands):
                            self.stack_machine.push(operand)

                elif top_element.element_type == ElementType.CONTROL:
                    self.stack_machine.pop()
                    control = top_element
                    print(f"Processing control instruction: {control.instruction}")
                    # Implement actual control logic if desired

                else:
                    # Just pop anything else
                    self.stack_machine.pop()

                continue

            # Mark the task as in progress
            self.stack_machine.update_element(
                next_task.element_id, status=TaskStatus.IN_PROGRESS
            )

            # Ask LLM for analysis
            analysis, success = self.analyze_task(next_task)
            if not success:
                # Task analysis failed -> retry or fail
                failure_reason = analysis.get("error", "Unknown analysis error")
                retry_count = next_task.retry_count + 1
                if retry_count <= next_task.max_retries:
                    self.stack_machine.update_element(
                        next_task.element_id,
                        status=TaskStatus.PENDING,
                        failure_reason=failure_reason,
                        retry_count=retry_count,
                    )
                    print(
                        f"Analysis failed for task: {next_task.description}. "
                        f"Retry {retry_count}/{next_task.max_retries}. Reason: {failure_reason}"
                    )
                else:
                    self.stack_machine.update_element(
                        next_task.element_id,
                        status=TaskStatus.FAILED,
                        failure_reason=failure_reason,
                    )
                    print(
                        f"Analysis final failure: {next_task.description}. "
                        f"Max retries exceeded. Reason: {failure_reason}"
                    )
                continue

            # Handle analysis results
            if analysis["task_type"] == "decomposition":
                # Mark current task as decomposition
                self.stack_machine.update_element(
                    next_task.element_id,
                    type=TaskType.DECOMPOSITION,
                    status=TaskStatus.COMPLETED,
                )

                # Create subtasks
                for subtask_info in analysis.get("subtasks", []):
                    sub_desc = subtask_info["description"]
                    sub_deps = subtask_info.get("dependencies", [])
                    subtask_id = self.stack_machine.create_task(
                        sub_desc,
                        dependencies=sub_deps,
                    )
                    self.stack_machine.push(self.stack_machine.get_element(subtask_id))

                print(f"Decomposed task: {next_task.description}")

            elif analysis["task_type"] == "execution":
                self.stack_machine.update_element(
                    next_task.element_id,
                    type=TaskType.EXECUTION,
                    tool_name=analysis["tool"],
                    tool_args=analysis.get("tool_args", {}),
                )

                # Execute
                result = self.execute_task(next_task)
                if "error" in result:
                    # Retry or final fail
                    failure_reason = f"Execution error: {result['error']}"
                    retry_count = next_task.retry_count + 1
                    if retry_count <= next_task.max_retries:
                        self.stack_machine.update_element(
                            next_task.element_id,
                            status=TaskStatus.PENDING,
                            failure_reason=failure_reason,
                            retry_count=retry_count,
                        )
                        print(
                            f"Task failed: {next_task.description}. "
                            f"Retry {retry_count}/{next_task.max_retries}. Reason: {failure_reason}"
                        )
                    else:
                        self.stack_machine.update_element(
                            next_task.element_id,
                            result=result,
                            status=TaskStatus.FAILED,
                            failure_reason=failure_reason,
                        )
                        print(
                            f"Task final failure: {next_task.description}. "
                            f"Max retries exceeded. Reason: {failure_reason}"
                        )
                else:
                    # Success
                    self.stack_machine.update_element(
                        next_task.element_id,
                        result=result,
                        status=TaskStatus.COMPLETED,
                    )
                    print(f"Task success: {next_task.description}")

                    # Create an operand for subsequent tasks if needed
                    operand_id = self.stack_machine.create_operand_from_task(
                        next_task.element_id
                    )
                    self.stack_machine.push(self.stack_machine.get_element(operand_id))
            elif analysis["task_type"] == "response":
                # Mark current task as response
                self.stack_machine.update_element(
                    next_task.element_id,
                    type=TaskType.RESPONSE,
                    result=analysis["response"],
                    status=TaskStatus.COMPLETED,
                )
                print(f"Responded to task: {next_task.description}")

        # Since there's no more tree structure, we just dump all final tasks
        # that ended in COMPLETED as the "results."
        final_results = []
        for elem_id, elem in self.stack_machine.elements.items():
            if elem.element_type == ElementType.TASK:
                task = elem
                if task.status == TaskStatus.COMPLETED:
                    final_results.append(
                        {
                            "task": task.description,
                            "result": task.result,
                            "status": task.status,
                        }
                    )

        return final_results


# Sample tool functions
def echo(message: str) -> Dict:
    return {"message": message}


def add(a: int, b: int) -> Dict:
    return {"result": a + b}


def fetch_web_content(url: str) -> Dict:
    try:
        response = requests.get(url)
        response.raise_for_status()
        # truncate for brevity
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


def compose_results(results: List[Dict], template: str = None) -> Dict:
    """Combine results from multiple tasks"""
    combined = {"combined_results": results}
    if template:
        formatted = template
        for i, r in enumerate(results):
            formatted = formatted.replace(f"{{result{i}}}", str(r))
        combined["formatted_result"] = formatted
    return combined


def main():
    # Initialize LLM client
    llm_client = LLMClient(api_url="http://localhost:11434", model="phi4-mini:latest")

    # Initialize tool registry and register sample tools
    tool_registry = ToolRegistry()
    tool_registry.register_tool(
        "echo", echo, 'echo(message: str) -> Dict["message": str]'
    )
    tool_registry.register_tool(
        "add", add, 'add(a: int, b: int) -> Dict["result": int]'
    )
    tool_registry.register_tool(
        "fetch_web_content",
        fetch_web_content,
        'fetch_web_content(url: str) -> Dict["status": str, "content": str]',
    )
    tool_registry.register_tool(
        "parse_html",
        parse_html,
        'parse_html(html_content: str, selector: str) -> Dict["status": str, "elements": List[str], "count": int]',
    )
    tool_registry.register_tool(
        "compose_results",
        compose_results,
        'compose_results(results: List[Dict], template: str) -> Dict["combined_results": List[Dict], "formatted_result": str]',
    )

    # Create the agent
    agent = StackMachineAgent(llm_client, tool_registry)

    # Example initial task (can be any instruction to the agent)
    initial_task_description = "파이썬 라이브러리 정보를 찾기 위해 Python.org와 PyPi.org를 조사하고 결과를 요약해줘"

    # Run the agent
    final_results = agent.run(initial_task_description)

    print("\n=== Final Completed Tasks ===")
    print(json.dumps(final_results, indent=2, ensure_ascii=False, cls=EnumEncoder))


if __name__ == "__main__":
    main()
