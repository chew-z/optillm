# Codanna CLI

## Using Codanna CLI for effective codebase search

Codanna is a CLI tool `codanna mcp` that gives the ability to search through codebase — understanding functions, tracing relationships, and finding implementations with surgical precision. Whenever analyzing codebase or searching through the project you MUST use Codanna tools first before using `grep` or `ripgrep`.

| Command                                             | Description                                   |
| --------------------------------------------------- | --------------------------------------------- |
| `codanna mcp semantic_search_with_context <NAME>`   | Search for symbols using full-text search     |
| `codanna mcp find_symbol name:<SYMBOL>`             | Show comprehensive information about a symbol |
| `codanna mcp get_calls function_name:<FUNCTION>`    | Show what functions a given function calls    |
| `codanna mcp find_callers function_name:<FUNCTION>` | Show what functions call a given function     |
| `codanna mcp analyze_impact symbol_name:<TRAIT>`    | Show what types implement a trait             |
| `codanna mcp search_symbols query:<QUERY>`          | Full-text search with fuzzy matching          |

You MUST always start with `codanna mcp semantic_search_with_context` and then follow up with other commands

## Follow the flow of Codanna system instructions

Codanna's guidance is model‑facing. Each tool response includes instructions for the use of tools in next steps like: `Focus on critical paths`, `investigate specific callers with 'find_symbol'`, `Consider 'analyze_impact' for complete dependency graph`.

Guidance is embedded directly in output text. Watch for phrases like "Consider...", "Try...", "This might be...". NEVER ignore the guidance. The tools know the codebase structure. You MUST always obtain these instructions to get the best results.

When in doubt, you MUST obtain suggestions directly using json output:

```bash
# Get suggestions for the next steps of the analysis
codanna mcp analyze_impact "QUERY" --json | jq -r '.system_message'
```

### Parameters Reference

| Tool                           | Parameters                            |
| ------------------------------ | ------------------------------------- |
| `find_symbol`                  | `name` (required)                     |
| `search_symbols`               | `query`, `limit`, `kind`, `module`    |
| `semantic_search_docs`         | `query`, `limit`, `threshold`, `lang` |
| `semantic_search_with_context` | `query`, `limit`, `threshold`, `lang` |
| `get_calls`                    | `function_name`                       |
| `find_callers`                 | `function_name`                       |
| `analyze_impact`               | `symbol_name`, `max_depth`            |
| `get_index_info`               | None                                  |

### Language Filtering

Semantic search tools support language filtering to reduce noise:

```bash
# Search only in Python code
codanna mcp semantic_search_docs query:"authentication" lang:python limit:5

# Search only in Go code
codanna mcp semantic_search_with_context query:"parse config" lang:go limit:3
```

## Advanced use of Codanna with `--json` flag and Unix pipe

All commands support `--json` flag for structured output (exit code 3 when not found). However, `--json` output is too verbose, so you MUST use commands without `--json` by default.
Only when you need structured data for further processing use `--json` flag and `jq` to filter and format the output. Here are some useful examples:

```bash
# Show start–end lines for each symbol
codanna mcp semantic_search_with_context query:"QUERY" --json \
| jq -r '.data[]? | "\(.symbol.name): \(.symbol.module_path) \(.symbol.range.start_line)-\(.symbol.range.end_line)"'

# Extract unique file paths from search results
codanna mcp semantic_search_with_context query:"QUERY" --json \
| jq -r '.data[]? | (.file_path // .symbol.file_path) | select(.)' \
| sort -u

# Show symbol names, their scope, file, and signature
codanna mcp semantic_search_with_context "QUERY" --json | \
  jq -r '.data[] | "\(.symbol.name) (\(.symbol.scope_context // "")) - \(.file_path)\n  \(.symbol.signature // "")"'
```
