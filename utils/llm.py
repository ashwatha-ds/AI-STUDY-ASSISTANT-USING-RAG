from groq import Groq
import json
import re


def _build_history_str(history: list) -> str:
    """Convert message history to a readable string for context."""
    lines = []
    for msg in history:
        role = "Student" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines[-10:])  # last 5 exchanges


def ask_groq(
    question: str,
    context: str | None,
    history: list,
    api_key: str,
    mode: str = "rag",
) -> str:
    """
    Query Groq LLM.
    - mode='rag'     : answer strictly from provided context
    - mode='general' : use general knowledge (hybrid fallback)
    """
    client = Groq(api_key=api_key)
    history_str = _build_history_str(history[:-1])  # exclude current question

    if mode == "rag":
        system_prompt = """You are StudyMind AI, an expert study assistant.
Answer the student's question ONLY using the provided document context.
Be clear, concise, and educational. Use bullet points when listing multiple points.
If the context doesn't fully answer the question, say so honestly.
Always cite which part of the document supports your answer."""

        user_content = f"""Previous conversation:
{history_str}

Document Context:
{context}

Student Question: {question}

Answer based on the document context above:"""

    else:
        system_prompt = """You are StudyMind AI, an expert study assistant and tutor.
The student's question goes beyond their uploaded documents.
Answer using your general knowledge in a clear, educational, and engaging way.
Use examples, analogies, and bullet points where helpful.
Note briefly that this answer comes from general AI knowledge."""

        user_content = f"""Previous conversation:
{history_str}

Student Question: {question}

Provide a helpful, educational answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.4,
        max_tokens=1024,
    )

    return response.choices[0].message.content


def summarize_document(text: str, style: str, api_key: str) -> str:
    """Generate a structured summary of document text."""
    client = Groq(api_key=api_key)

    style_instructions = {
        "Concise (bullet points)": "Summarize in concise bullet points. Group by topic. Max 15 bullets.",
        "Detailed (paragraph)": "Write a detailed 3-5 paragraph summary covering all major topics.",
        "Key Concepts Only": "Extract only the KEY CONCEPTS and DEFINITIONS as a numbered list. Focus on terms a student must know.",
    }

    instruction = style_instructions.get(style, style_instructions["Concise (bullet points)"])

    prompt = f"""You are an expert academic summarizer.

{instruction}

Document text:
{text[:12000]}

Provide the summary now:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1500,
    )

    return response.choices[0].message.content


def generate_quiz(text: str, num_questions: int, difficulty: str, api_key: str) -> dict:
    """Generate MCQ quiz questions from document text."""
    client = Groq(api_key=api_key)

    difficulty_guide = {
        "Easy": "straightforward recall questions with obvious wrong answers",
        "Medium": "application-level questions requiring understanding",
        "Hard": "analytical questions requiring deep understanding and inference",
    }

    prompt = f"""You are an expert quiz creator for students.
Generate exactly {num_questions} multiple choice questions at {difficulty} difficulty ({difficulty_guide[difficulty]}).

Rules:
- Each question must have exactly 4 options (A, B, C, D format in the text, but return them as a plain list)
- One clearly correct answer
- Include a brief explanation for the correct answer
- Base questions ONLY on the provided text

Return ONLY a valid JSON object with this exact structure:
{{
  "questions": [
    {{
      "question": "Question text here?",
      "options": ["Option A text", "Option B text", "Option C text", "Option D text"],
      "answer": "Option A text",
      "explanation": "Brief explanation why this is correct"
    }}
  ]
}}

Document text:
{text[:10000]}

Return only the JSON, no other text:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=2000,
    )

    raw = response.choices[0].message.content.strip()

    
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from response
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return {"error": "Could not parse quiz. Try again or use a different document section."}
