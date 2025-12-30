import anthropic
import json
from typing import List, Optional, Dict, Any, Tuple


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Maximum number of sequential tool calling rounds
    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant named Alfie, specialized in course materials and educational content with access to tools for searching course content and retrieving course outlines.

Available Tools:
1. **search_course_content**: Search course materials for specific content or detailed educational materials
2. **get_course_outline**: Get course structure including title, link, and complete lesson list with numbers and titles

Tool Usage Guidelines:
- Use **get_course_outline** for questions about:
  - Course structure, outline, or overview
  - What lessons are in a course
  - Course table of contents
  - Questions like "what does this course cover", "list the lessons", "show me the course outline"
  - ANY question asking about what a course contains or its structure
- Use **search_course_content** for questions about:
  - Specific course content or topics within lessons
  - Detailed educational materials or explanations
  - Questions about HOW to do something or WHAT something means
- You may use tools sequentially if needed (e.g., get outline first, then search content)
- Use additional tool calls only when the first result is insufficient
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course outline questions**: Use get_course_outline tool. In your response, you MUST include:
  1. The course title
  2. The course link
  3. ALL lessons with their lesson numbers and titles exactly as returned by the tool
  Do NOT summarize or skip any lessons. Do NOT add descriptions - just list lesson numbers and titles.
- **Course content questions**: Use search_course_content, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Start every answer with "Hi I'm Alfie, your course assistant!"
 - Do not mention "based on the search results" or "based on the outline"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports up to MAX_TOOL_ROUNDS sequential tool calls.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Initialize messages
        messages = [{"role": "user", "content": query}]

        # Sequential tool calling loop
        for round_num in range(self.MAX_TOOL_ROUNDS):
            print(f"\n=== Tool Round {round_num + 1}/{self.MAX_TOOL_ROUNDS} ===")

            # Prepare API call parameters
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content
            }

            # Add tools if available
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            # Get response from Claude
            print(f"Calling API with {len(messages)} messages...")
            response = self.client.messages.create(**api_params)
            print(f"Response stop_reason: {response.stop_reason}")

            # If no tool use, return the response
            if response.stop_reason != "tool_use" or not tool_manager:
                print(f"  No tool use requested, returning response")
                return self._extract_text_response(response)

            # Execute tools and update messages
            messages, error_occurred = self._execute_tools_and_update_messages(
                response, messages, tool_manager
            )

            # If tool execution failed critically, break and get final response
            if error_occurred:
                break

        # After loop ends (max rounds reached or error), make final call without tools
        print(f"\n=== Final API Call (no tools) ===")
        print(f"Total messages: {len(messages)}")
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
        }

        print("Final Params:")
        print(json.dumps(final_params, indent=2, default=str))

        final_response = self.client.messages.create(**final_params)
        print("Final Response:")
        print(final_response)
        return self._extract_text_response(final_response)

    def _extract_text_response(self, response) -> str:
        """
        Extract text content from an API response.

        Args:
            response: The API response object

        Returns:
            Text content from the response
        """
        for content_block in response.content:
            if hasattr(content_block, 'text'):
                return content_block.text
        return ""

    def _execute_tools_and_update_messages(
        self,
        response,
        messages: List[Dict],
        tool_manager
    ) -> Tuple[List[Dict], bool]:
        """
        Execute tool calls from response and update message history.

        Args:
            response: The API response containing tool use requests
            messages: Current message history
            tool_manager: Manager to execute tools

        Returns:
            Tuple of (updated messages, error_occurred flag)
        """
        error_occurred = False

        # Add AI's tool use response to messages
        messages.append({"role": "assistant", "content": response.content})

        # Execute all tool calls and collect results
        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                print(f"  Executing tool: {content_block.name}")
                print(f"  Tool input: {content_block.input}")
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name,
                        **content_block.input
                    )
                    print(f"  Tool result length: {len(tool_result)} chars")
                except Exception as e:
                    tool_result = f"Error executing tool: {str(e)}"
                    error_occurred = True
                    print(f"  Tool error: {e}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })

        # Add tool results as user message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        return messages, error_occurred
