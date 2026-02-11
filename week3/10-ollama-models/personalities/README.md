# Personality Modelfiles

Three custom Ollama personalities that demonstrate how system prompts and parameter tuning change model behavior dramatically.

## Available Personalities

| Personality | Behavior | Temperature |
|---|---|---|
| **assumption-auditor** | Surfaces hidden assumptions before answering, rates each [Strong/Medium/Weak] | 0.3 |
| **executive-briefer** | Delivers exactly 3 one-sentence bullets, no hedging | 0.4 |
| **overcautious-gatekeeper** | Refuses to answer until it has enough information, lists what's missing | 0.2 |

## Usage

1. **Create a personality model:**
   ```bash
   cd week3/10-ollama-models/personalities/assumption-auditor
   ollama create assumption-auditor -f Modelfile
   ```

2. **Run it:**
   ```bash
   ollama run assumption-auditor
   ```

3. **Try the same prompt with all three** to see how personality shapes output:
   ```bash
   # Same question, very different answers:
   echo "Should we migrate to microservices?" | ollama run assumption-auditor
   echo "Should we migrate to microservices?" | ollama run executive-briefer
   echo "Should we migrate to microservices?" | ollama run overcautious-gatekeeper
   ```

## Creating All Three at Once

```bash
cd week3/10-ollama-models/personalities
ollama create assumption-auditor -f assumption-auditor/Modelfile
ollama create executive-briefer -f executive-briefer/Modelfile
ollama create overcautious-gatekeeper -f overcautious-gatekeeper/Modelfile
```

## Key Takeaway

These examples show that the same base model (`llama3.2`) produces radically different outputs depending on the system prompt and temperature. This is a core LLM skill: designing personas through prompt engineering rather than fine-tuning.
