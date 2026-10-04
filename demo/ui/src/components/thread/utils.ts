import type { Message } from "@langchain/langgraph-sdk";

/**
 * Extracts a string summary from a message's content, supporting multimodal (text, image, file, etc.).
 * - If text is present, returns the joined text.
 * - If not, returns a label for the first non-text modality (e.g., 'Image', 'Other').
 * - If unknown, returns 'Multimodal message'.
 */
export function getContentString(content: Message["content"]): string {
  if (typeof content === "string") return content;
  const texts = content
    .filter((c): c is { type: "text"; text: string } => c.type === "text")
    .map((c) => c.text);
  return texts.join(" ");
}

/**
 * Terminal agent responses are stored as JSON in an AI message.  Show their
 * user-facing message in chat, while retaining the structured status in state.
 */
export function getAssistantDisplayContent(content: Message["content"]): string {
  const text = getContentString(content);
  try {
    const value: unknown = JSON.parse(text);
    if (
      typeof value === "object" &&
      value !== null &&
      "message" in value &&
      typeof value.message === "string" &&
      "status" in value &&
      (value.status === "done" || value.status === "needs_user")
    ) {
      return value.message;
    }
  } catch {
    // Regular assistant text is not JSON.
  }
  return text;
}
