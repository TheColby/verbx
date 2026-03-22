# Structured JSON Schemas

This document provides JSON Schema (Draft 2020-12) references for the two primary structured payloads used in `0.7.x`: batch render manifests and automation files.

## 1) Batch Render Manifest Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://verbx.dev/schema/batch-render-manifest-v1.json",
  "title": "verbx batch render manifest",
  "type": "object",
  "additionalProperties": false,
  "required": ["jobs"],
  "properties": {
    "jobs": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["infile", "outfile"],
        "properties": {
          "infile": { "type": "string", "minLength": 1 },
          "outfile": { "type": "string", "minLength": 1 },
          "options": {
            "type": "object",
            "description": "RenderConfig-like key/value options accepted by verbx batch render"
          }
        }
      }
    }
  }
}
```

## 2) Automation File Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://verbx.dev/schema/automation-lanes-v1.json",
  "title": "verbx automation lanes",
  "type": "object",
  "required": ["lanes"],
  "additionalProperties": true,
  "properties": {
    "mode": { "type": "string", "enum": ["sample", "block"] },
    "block_ms": { "type": "number", "exclusiveMinimum": 0 },
    "lanes": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "required": ["target", "type"],
        "properties": {
          "target": { "type": "string", "minLength": 1 },
          "type": {
            "type": "string",
            "enum": ["breakpoints", "lfo", "envelope", "value", "feature"]
          },
          "interp": {
            "type": "string",
            "enum": ["linear", "hold", "step", "smooth", "smoothstep", "exp", "exponential"]
          },
          "points": {
            "type": "array",
            "items": {
              "type": "object",
              "required": ["time", "value"],
              "properties": {
                "time": { "type": "number", "minimum": 0 },
                "value": { "type": "number" }
              },
              "additionalProperties": true
            }
          }
        },
        "additionalProperties": true
      }
    }
  }
}
```

## Notes

- `jobs[].options` is intentionally open-ended to track `RenderConfig` growth without forcing schema churn.
- Lane payloads accept additional fields to support advanced lane types (for example, feature-vector controls).
