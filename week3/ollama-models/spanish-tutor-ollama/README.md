# Spanish Tutor - Custom Ollama Model

A custom Ollama model configured as an interactive Spanish language tutor named "María".

## Prerequisites

- [Ollama](https://ollama.ai/) installed and running
- Base model pulled: `ollama pull deepseek-r1`

## Setup

1. **Create the custom model:**
   ```bash
   cd week3/spanish-tutor-ollama
   ollama create spanish-tutor -f Modelfile
   ```

2. **Verify the model was created:**
   ```bash
   ollama list
   ```
   You should see `spanish-tutor` in the list.

## Usage

### Interactive Chat
```bash
ollama run spanish-tutor
```

### Example Prompts

**For Beginners:**
```
I'm a complete beginner. Can you teach me how to introduce myself in Spanish?
```

**For Grammar Help:**
```
Can you explain the difference between ser and estar?
```

**For Conversation Practice:**
```
Let's practice ordering food at a restaurant. You be the waiter.
```

**For Corrections:**
```
Is this sentence correct? "Yo soy tiene hambre"
```

## Features

- **Adaptive difficulty** - Adjusts teaching style based on your level (A1-C2)
- **Grammar explanations** - Clear rules with examples and memory tricks
- **Conversation practice** - Role-play real-world scenarios
- **Cultural context** - Learn customs alongside language
- **Error correction** - Supportive feedback that explains the "why"

## Customization

Edit the `Modelfile` to adjust:

- **Base model**: Change `FROM deepseek-r1:latest` to another model
- **Temperature**: Higher (0.8-1.0) for creative responses, lower (0.3-0.5) for more consistent teaching
- **System prompt**: Modify the tutor's personality or focus areas

### Rebuilding After Changes
```bash
ollama rm spanish-tutor
ollama create spanish-tutor -f Modelfile
```

## Tips for Learning

1. **Start with your level** - Tell María if you're a beginner, intermediate, or advanced
2. **Practice regularly** - Short daily sessions beat long weekly ones
3. **Make mistakes** - María will correct you kindly and explain why
4. **Ask "why"** - Understanding grammar rules helps retention
5. **Request scenarios** - Practice real situations you'll encounter
