import json
import re
import uuid
import requests
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable
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
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    result: Any = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict] = None
    failure_reason: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = field(
        default_factory=list
    )  # IDs of tasks this task depends on

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
                f"Error executing tool '{name}': {e}; Args: {args}, Documenation: {self.tools_docs.get(name)}"
            )


# Stack Machine for the Agent
class StackMachine:
    def __init__(self):
        self.stack = deque()  # Operating stack
        self.elements = (
            {}
        )  # element_id -> StackElement (for all elements that have been in the stack)
        self.element_hashes = {}  # element_hash -> element_id (for duplicate detection)
        self.root_tasks = []  # List of root task IDs

    def push(self, element: StackElement) -> None:
        """Push an element onto the stack"""
        self.elements[element.element_id] = element
        self.stack.append(element.element_id)

        # For tasks, store a hash for duplicate detection
        if element.element_type == ElementType.TASK:
            task = element
            self.element_hashes[task.hash] = task.element_id

            # Add to root tasks if it has no parent
            if not task.parent_id:
                self.root_tasks.append(task.element_id)

    def pop(self) -> Optional[StackElement]:
        """Pop and return the top element from the stack"""
        if not self.stack:
            return None
        element_id = self.stack.pop()
        return self.elements.get(element_id)

    def peek(self) -> Optional[StackElement]:
        """Look at the top element without removing it"""
        if not self.stack:
            return None
        element_id = self.stack[-1]
        return self.elements.get(element_id)

    def size(self) -> int:
        """Get current stack size"""
        return len(self.stack)

    def is_empty(self) -> bool:
        """Check if the stack is empty"""
        return len(self.stack) == 0

    def get_element(self, element_id: str) -> Optional[StackElement]:
        """Get an element by ID"""
        return self.elements.get(element_id)

    def update_element(self, element_id: str, **kwargs) -> bool:
        """Update an element by ID with provided attributes"""
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
        parent_id: Optional[str] = None,
        dependencies: List[str] = None,
        max_retries: int = 3,
    ) -> str:
        """Create a new task (with duplicate detection and reuse)"""
        # Generate hash from task description
        task_hash = hashlib.md5(description.lower().strip().encode()).hexdigest()

        # Check if this task already exists
        if task_hash in self.element_hashes:
            existing_task_id = self.element_hashes[task_hash]
            existing_task = self.elements.get(existing_task_id)

            # If there's a parent task, connect as child
            if parent_id:
                parent_task = self.elements.get(parent_id)
                if parent_task and existing_task_id not in parent_task.children_ids:
                    parent_task.children_ids.append(existing_task_id)
                    existing_task.parent_id = parent_id

            # Add any new dependencies
            if dependencies:
                for dep_id in dependencies:
                    if dep_id not in existing_task.dependencies:
                        existing_task.dependencies.append(dep_id)

            return existing_task_id

        # Create new task
        task_id = str(uuid.uuid4())
        task = Task(
            element_id=task_id,
            element_type=ElementType.TASK,
            description=description,
            status=TaskStatus.PENDING,
            parent_id=parent_id,
            dependencies=dependencies or [],
            max_retries=max_retries,
        )

        # Set parent-child relationship
        if parent_id:
            parent_task = self.elements.get(parent_id)
            if parent_task:
                parent_task.children_ids.append(task_id)

        # Add to elements and store hash
        self.elements[task_id] = task
        self.element_hashes[task_hash] = task_id

        return task_id

    def get_next_executable_task(self) -> Optional[Task]:
        """Get the next executable task (considering dependencies)"""
        for element_id in self.stack:
            element = self.elements.get(element_id)
            if not element or element.element_type != ElementType.TASK:
                continue

            task = element
            if task.status != TaskStatus.PENDING:
                continue

            # Check if all dependencies are satisfied
            dependencies_met = True
            for dep_id in task.dependencies:
                dep_task = self.elements.get(dep_id)
                if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                    dependencies_met = False
                    break

            # Check if all children are completed (for decomposition tasks)
            children_completed = True
            if task.children_ids:
                for child_id in task.children_ids:
                    child_task = self.elements.get(child_id)
                    if not child_task or child_task.status != TaskStatus.COMPLETED:
                        children_completed = False
                        break

            if dependencies_met and children_completed:
                return task

        return None

    def create_operand_from_task(self, task_id: str) -> str:
        """Create an operand from a task's result"""
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

    def get_task_tree(self) -> Dict:
        """Return the task tree structure"""

        def build_tree(element_id):
            element = self.elements[element_id]

            if element.element_type == ElementType.TASK:
                task = element
                tree = asdict(task)

                if task.children_ids:
                    tree["children"] = [
                        build_tree(child_id) for child_id in task.children_ids
                    ]

                return tree
            else:
                return asdict(element)

        return [build_tree(root_id) for root_id in self.root_tasks]

    def get_similar_tasks(self, description: str, limit: int = 5) -> List[Task]:
        """Find similar tasks (simple keyword matching)"""
        keywords = set(description.lower().split())
        task_scores = []

        for element_id, element in self.elements.items():
            if element.element_type == ElementType.TASK:
                task = element
                if task.status == TaskStatus.COMPLETED:
                    task_keywords = set(task.description.lower().split())
                    common_words = keywords.intersection(task_keywords)
                    score = len(common_words) / max(len(keywords), len(task_keywords))

                    if score > 0.2:  # Only tasks with similarity above threshold
                        task_scores.append((score, task))

        # Sort by similarity and return top N
        similar_tasks = [
            task
            for _, task in sorted(task_scores, key=lambda x: x[0], reverse=True)[:limit]
        ]
        return similar_tasks


# Agent class using stack machine architecture
class StackMachineAgent:
    def __init__(self, llm_client: LLMClient, tool_registry: ToolRegistry):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.stack_machine = StackMachine()

    def _create_task_analysis_prompt(self, task: Task) -> str:
        """Create prompt for task analysis"""
        # Available tools
        available_tools = list(self.tool_registry.tools.keys())
        available_tools_with_docs = (
            f"{tool}: {self.tool_registry.tools_docs.get(tool, '')}"
            for tool in available_tools
        )

        # Similar tasks
        similar_tasks = self.stack_machine.get_similar_tasks(task.description)

        # Extract decomposition results from similar tasks
        reusable_subtasks = []
        for similar_task in similar_tasks:
            if similar_task.type == TaskType.DECOMPOSITION:
                child_tasks = [
                    self.stack_machine.get_element(child_id).description
                    for child_id in similar_task.children_ids
                ]
                reusable_subtasks.extend(child_tasks)

        # Remove duplicates and limit to 10
        reusable_subtasks = list(set(reusable_subtasks))[:10]
        reusable_subtasks_text = "\n".join(
            [f"- {subtask}" for subtask in reusable_subtasks]
        )

        # Add information about task dependencies and their results
        dependency_info = ""
        if task.dependencies:
            dependency_results = []
            for dep_id in task.dependencies:
                dep_task = self.stack_machine.get_element(dep_id)
                if dep_task and dep_task.status == TaskStatus.COMPLETED:
                    dependency_results.append(
                        f"Dependency: {dep_task.description}\nResult: {dep_task.result}"
                    )

            if dependency_results:
                dependency_info = (
                    "Previous task results (available as context):\n"
                    + "\n".join(dependency_results)
                )

        # Add previous failure information
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
1. Use an available tool to execute the task directly.
2. Decompose the task into two subtasks for further analysis.

Task: {task.description}

Available tools: {', '.join(available_tools_with_docs)}
{dependency_info}
{failure_info}

Previously generated subtasks (available for reuse):
{reusable_subtasks_text if reusable_subtasks else ""}

Reuse guidelines:
- Minimize duplication by reusing existing subtasks whenever possible.
- If a new subtask is similar to an existing one, use the exact same description.
- When creating new subtasks, write them in a general format to allow for future reuse.
- You can specify dependencies between subtasks to ensure proper execution order.

Provide your response in the following JSON format:

```
{{
    "task_type": "decomposition or execution",
    "reasoning": "Your reasoning for the decision",
    "subtasks": [
        {{
            "description": "Subtask 1",
            "dependencies": [] // IDs of other subtasks this depends on (if any)
        }},
        {{
            "description": "Subtask 2",
            "dependencies": [] // IDs of other subtasks this depends on (if any)
        }}
    ], // Only for task_type = decomposition
    "tool": "Tool name", // Only for task_type = execution
    "tool_args": {{}} // Only for task_type = execution
}}
```

Ensure the response is in a valid JSON format. Always add a comma "between" object fields.
Do not use comments, and `subtasks` must be a list of objects with description and dependencies fields.
When specifying tool arguments, use data from dependency results when appropriate.
Do not use placeholders—provide actual values. Whenever possible, use the given tools to complete the task.
"""
        return prompt

    def analyze_task(self, task: Task) -> Tuple[Dict, bool]:
        """Analyze a task using LLM"""
        prompt = self._create_task_analysis_prompt(task)
        response = self.llm_client.query(prompt)

        # Extract and parse JSON response
        try:
            # Remove think tokens
            response = re.sub(
                r"<think>.*?</think>", "", response, flags=re.DOTALL
            ).strip()

            # Handle JSON wrapped in code blocks
            if "```json" in response or "```" in response:
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            else:
                # Assume entire response is JSON
                response = response.strip()

            analysis = json.loads(response.strip())

            # Validate response
            if analysis.get("task_type") == "decomposition":
                if not analysis.get("subtasks") or not isinstance(
                    analysis["subtasks"], list
                ):
                    return {
                        "task_type": "execution",
                        "reasoning": "Subtasks missing or invalid",
                        "tool": "echo",
                        "tool_args": {
                            "message": "Task analysis failed: Invalid subtasks"
                        },
                        "error": "Task analysis failed: Subtasks missing or invalid",
                    }, False

                # Validate subtask format
                for subtask in analysis["subtasks"]:
                    if not isinstance(subtask, dict) or "description" not in subtask:
                        return {
                            "task_type": "execution",
                            "reasoning": "Subtask must have a description",
                            "tool": "echo",
                            "tool_args": {
                                "message": "Task analysis failed: Subtask missing description"
                            },
                            "error": "Task analysis failed: Subtask must have a description",
                        }, False

            elif analysis.get("task_type") == "execution":
                if not analysis.get("tool"):
                    return {
                        "task_type": "execution",
                        "reasoning": "Tool name not specified",
                        "tool": "echo",
                        "tool_args": {
                            "message": "Task analysis failed: Missing tool name"
                        },
                        "error": "Task analysis failed: Tool name not specified",
                    }, False

                if analysis.get("tool") not in self.tool_registry.tools:
                    return {
                        "task_type": "execution",
                        "reasoning": f"Tool '{analysis.get('tool')}' not registered",
                        "tool": "echo",
                        "tool_args": {
                            "message": f"Task analysis failed: Unknown tool '{analysis.get('tool')}'"
                        },
                        "error": f"Task analysis failed: Tool '{analysis.get('tool')}' not registered",
                    }, False

            else:
                return {
                    "task_type": "execution",
                    "reasoning": "Invalid task_type",
                    "tool": "echo",
                    "tool_args": {
                        "message": "Task analysis failed: task_type must be 'decomposition' or 'execution'"
                    },
                    "error": "Task analysis failed: task_type must be 'decomposition' or 'execution'",
                }, False

            return analysis, True

        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Original response: {response}")

            # Return default value
            return {
                "task_type": "execution",
                "reasoning": f"JSON parsing error: {e}",
                "tool": "echo",
                "tool_args": {
                    "message": f"Task analysis failed: JSON parsing error - {e}"
                },
                "error": f"Task analysis failed: JSON parsing error - {e}",
            }, False

    def execute_task(self, task: Task) -> Any:
        """Execute task using a tool"""
        if not task.tool_name or not task.tool_args:
            return {"error": "Tool information missing."}

        try:
            # Get dependency results if needed
            dependency_results = {}
            for dep_id in task.dependencies:
                dep_task = self.stack_machine.get_element(dep_id)
                if dep_task and dep_task.status == TaskStatus.COMPLETED:
                    dependency_results[dep_id] = dep_task.result

            # If tool_args contains references to dependency results, replace them
            tool_args = task.tool_args
            # (Here you could implement a method to inject dependency results if needed)

            result = self.tool_registry.execute_tool(task.tool_name, tool_args)
            return result
        except Exception as e:
            return {"error": str(e)}

    def run(self, initial_task: str):
        """Run the agent with stack machine architecture"""
        # Create initial task
        initial_task_id = self.stack_machine.create_task(initial_task)

        # Push initial task to stack
        initial_task = self.stack_machine.get_element(initial_task_id)
        self.stack_machine.push(initial_task)

        # Run until stack is empty
        while not self.stack_machine.is_empty():
            # Get next executable task
            next_task = self.stack_machine.get_next_executable_task()
            if not next_task:
                # If no executable task, pop the top element
                top_element = self.stack_machine.peek()

                if top_element.element_type == ElementType.TASK:
                    task = top_element

                    # If it's a completed task, convert it to an operand
                    if task.status == TaskStatus.COMPLETED:
                        self.stack_machine.pop()  # Remove the task

                        # Create an operand from the task result
                        operand_id = self.stack_machine.create_operand_from_task(
                            task.element_id
                        )
                        operand = self.stack_machine.get_element(operand_id)

                        # Push the operand
                        self.stack_machine.push(operand)
                        print(
                            f"Converted completed task to operand: {task.description}"
                        )
                    else:
                        # If task can't be executed and isn't completed, remove it
                        self.stack_machine.pop()
                        print(
                            f"Removing unexecutable task from stack: {task.description}"
                        )

                elif top_element.element_type == ElementType.OPERATOR:
                    # Process operator by finding operands and applying operation
                    operator = top_element
                    self.stack_machine.pop()  # Remove operator

                    # Get operands
                    operands = []
                    for _ in range(operator.operand_count):
                        if self.stack_machine.is_empty():
                            break
                        element = self.stack_machine.pop()
                        if element.element_type == ElementType.OPERAND:
                            operands.append(element)

                    # If we have enough operands, execute the operation
                    if len(operands) == operator.operand_count:
                        # Execute operation (would need implementation based on operator type)
                        result = {"message": f"Applied operator {operator.operation}"}

                        # Create new operand with result
                        operand_id = str(uuid.uuid4())
                        new_operand = Operand(
                            element_id=operand_id,
                            element_type=ElementType.OPERAND,
                            value=result,
                        )

                        self.stack_machine.elements[operand_id] = new_operand
                        self.stack_machine.push(new_operand)
                    else:
                        # Not enough operands, put back what we found
                        for operand in reversed(operands):
                            self.stack_machine.push(operand)

                elif top_element.element_type == ElementType.CONTROL:
                    # Handle control instructions
                    control = top_element
                    self.stack_machine.pop()

                    # Implementation would depend on the control instruction
                    if control.instruction == "RETURN":
                        # Return from current execution context
                        print(f"Executing control instruction: {control.instruction}")
                    elif control.instruction == "LOOP":
                        # Implement loop logic
                        print(f"Executing control instruction: {control.instruction}")

                else:
                    # For other elements, just pop
                    self.stack_machine.pop()

                continue

            # Mark task as in progress
            self.stack_machine.update_element(
                next_task.element_id, status=TaskStatus.IN_PROGRESS
            )

            # Analyze task
            analysis, success = self.analyze_task(next_task)

            if not success:
                # Handle analysis failure
                failure_reason = f"Task analysis failed: {analysis.get('error')}"
                retry_count = next_task.retry_count + 1

                if retry_count <= next_task.max_retries:
                    self.stack_machine.update_element(
                        next_task.element_id,
                        status=TaskStatus.PENDING,
                        failure_reason=failure_reason,
                        retry_count=retry_count,
                    )
                    print(
                        f"Task analysis failed: {next_task.description}. Retry {retry_count}/{next_task.max_retries} ({failure_reason})"
                    )
                else:
                    self.stack_machine.update_element(
                        next_task.element_id,
                        status=TaskStatus.FAILED,
                        failure_reason=failure_reason,
                    )
                    print(
                        f"Task analysis final failure: {next_task.description}. Max retries exceeded"
                    )
                continue

            task_type = analysis.get("task_type")

            if task_type == "decomposition":
                # Decompose task into subtasks
                self.stack_machine.update_element(
                    next_task.element_id, type=TaskType.DECOMPOSITION
                )

                # Create subtasks with dependencies
                subtasks = analysis.get("subtasks", [])
                subtask_ids = []

                for subtask_info in subtasks:
                    subtask_description = subtask_info.get("description")
                    subtask_dependencies = subtask_info.get("dependencies", [])

                    subtask_id = self.stack_machine.create_task(
                        subtask_description,
                        parent_id=next_task.element_id,
                        dependencies=subtask_dependencies,
                    )
                    subtask_ids.append(subtask_id)

                    # Push subtask onto stack (in reverse order to maintain execution order)
                    subtask = self.stack_machine.get_element(subtask_id)
                    self.stack_machine.push(subtask)

                # Mark current task as completed
                self.stack_machine.update_element(
                    next_task.element_id, status=TaskStatus.COMPLETED
                )
                print(
                    f"Task decomposed: {next_task.description} -> {len(subtasks)} subtasks"
                )

            elif task_type == "execution":
                # Execute task
                self.stack_machine.update_element(
                    next_task.element_id,
                    type=TaskType.EXECUTION,
                    tool_name=analysis.get("tool"),
                    tool_args=analysis.get("tool_args", {}),
                )

                # Execute tool
                result = self.execute_task(next_task)

                # Save result and mark task as completed/failed
                if "error" in result:
                    # Handle failure
                    failure_reason = f"Tool execution error: {result['error']}"
                    retry_count = next_task.retry_count + 1

                    if retry_count <= next_task.max_retries:
                        # Set status to PENDING for retry and record failure reason
                        self.stack_machine.update_element(
                            next_task.element_id,
                            status=TaskStatus.PENDING,
                            failure_reason=failure_reason,
                            retry_count=retry_count,
                        )
                        print(
                            f"Task failed: {next_task.description}. Retry {retry_count}/{next_task.max_retries}. Reason: {failure_reason}"
                        )
                    else:
                        # Max retries exceeded
                        self.stack_machine.update_element(
                            next_task.element_id,
                            result=result,
                            status=TaskStatus.FAILED,
                            failure_reason=failure_reason,
                        )
                        print(
                            f"Task final failure: {next_task.description}. Max retries exceeded. Reason: {failure_reason}"
                        )
                else:
                    # Handle success
                    self.stack_machine.update_element(
                        next_task.element_id, result=result, status=TaskStatus.COMPLETED
                    )
                    print(f"Task execution successful: {next_task.description}")

                    # Create an operand from the result
                    operand_id = self.stack_machine.create_operand_from_task(
                        next_task.element_id
                    )
                    operand = self.stack_machine.get_element(operand_id)

                    # Push operand to stack for potential use by other tasks
                    self.stack_machine.push(operand)

        # Return final result
        return self.stack_machine.get_task_tree()


# Sample tool functions
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


# Compose function to combine results from multiple tasks
def compose_results(results: List[Dict], template: str = None) -> Dict:
    """Combine results from multiple tasks"""
    combined = {"combined_results": results}

    if template:
        # Apply template to format results
        # This is a simple implementation - could be extended with actual templating
        formatted = template
        for i, result in enumerate(results):
            formatted = formatted.replace(f"{{result{i}}}", str(result))

        combined["formatted_result"] = formatted

    return combined


def main():
    # Initialize LLM client (e.g., using Ollama API)
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

    # Initialize agent with LLM and tools
    agent = StackMachineAgent(llm_client, tool_registry)

    # Initial high-level task
    initial_task = "파이썬 라이브러리 정보를 찾기 위해 Python.org와 PyPi.org를 조사하고 결과를 요약해줘"

    # Run the agent
    final_result = agent.run(initial_task)

    # Print final task tree result
    print("\n=== 최종 태스크 트리 ===")
    print(json.dumps(final_result, indent=2, ensure_ascii=False, cls=EnumEncoder))


if __name__ == "__main__":
    main()
