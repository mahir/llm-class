# Socratic Tutor - Custom Ollama Model

A tutor that guides learning through questions rather than direct answers, based on the Socratic method of teaching.

## The Socratic Method

Instead of providing answers directly, this tutor:
- Asks guiding questions to help you discover the answer
- Builds on your existing knowledge
- Breaks complex problems into smaller pieces
- Helps you develop independent problem-solving skills

## Setup

```bash
# Pull the base model
ollama pull phi3.5

# Create the tutor
cd week3/socratic-tutor-ollama
ollama create socratic-tutor -f Modelfile

# Start learning
ollama run socratic-tutor
```

## Example Interaction

**You:** How does recursion work?

**Tutor:** Great question! Before we dive in—have you ever seen something that contains a smaller version of itself? Like a mirror reflecting another mirror, or a folder containing folders?

**You:** Yeah, like Russian nesting dolls?

**Tutor:** Perfect analogy! Each doll contains a smaller doll inside. Now, what would happen if you wanted to count all the dolls? How would you describe that process?

**You:** Open each one, count it, then check if there's another inside...

**Tutor:** Exactly! And when do you stop?

**You:** When there's no more dolls inside—the smallest one.

**Tutor:** You just described recursion! A function that calls itself, with a "base case" that tells it when to stop. Can you think of how you'd write a function to count down from 5 to 1 using this idea?

## Use Cases

- Learning programming concepts
- Understanding math problems
- Developing critical thinking
- Preparing for technical interviews
- Any subject where deep understanding matters

## Why Socratic Learning?

Research shows that actively discovering answers leads to:
- Better retention than passive learning
- Stronger problem-solving skills
- Deeper conceptual understanding
- Increased confidence in tackling new problems

## Customization

In the Modelfile, you can adjust:
- **Temperature**: Lower (0.4) for more focused questioning, higher (0.8) for more creative analogies
- **Subject focus**: Modify the system prompt to specialize in specific topics

## See Also

- `spanish-tutor-ollama/` - Direct instruction example
- Ollama Modelfile docs: https://github.com/ollama/ollama/blob/main/docs/modelfile.md
