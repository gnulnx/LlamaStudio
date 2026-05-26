import ast
import json
import re
import unittest


def parse_args_str(args_str: str) -> dict:
    args_str = args_str.strip()
    if args_str.startswith("{") and args_str.endswith("}"):
        try:
            return json.loads(args_str)
        except Exception:
            pass
        try:
            import yaml

            res = yaml.safe_load(args_str)
            if isinstance(res, dict):
                return res
        except Exception:
            pass

        inner = args_str[1:-1].strip()
        args = {}
        matches = re.finditer(
            r'(?:"([^"]+)"|\'([^\']+)\'|(\w+))\s*:\s*(?:"([^"]*)"|\'([^\']*)\'|([^\n,{}]+))', inner
        )
        for m in matches:
            key = m.group(1) or m.group(2) or m.group(3)
            val = m.group(4) or m.group(5) or m.group(6)
            if val is not None:
                val = val.strip()
                if (val.startswith('"') and val.endswith('"')) or (
                    val.startswith("'") and val.endswith("'")
                ):
                    val = val[1:-1]
                args[key] = val
        return args
    else:
        try:
            tree = ast.parse(f"dummy({args_str})")
            args = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    for kw in node.keywords:
                        if hasattr(kw.value, "value"):
                            args[kw.arg] = kw.value.value
                        elif hasattr(kw.value, "s"):
                            args[kw.arg] = kw.value.s
                        elif hasattr(kw.value, "n"):
                            args[kw.arg] = kw.value.n
            return args
        except Exception:
            args = {}
            matches = re.findall(r'(\w+)\s*=\s*(?:"(.*?)"|\'(.*?)\')', args_str, re.DOTALL)
            for m in matches:
                kw = m[0]
                val = m[1] if m[1] else m[2]
                args[kw] = val
            return args


def process_message_content(content_str: str) -> tuple[list[dict], str]:
    # 1. Extract hallucinated response block if present (do this FIRST while im_end tags are still there)
    content_str = re.sub(
        r"<\|im_start\|>response:.*?(?:<\|im_end\|>|$)", "", content_str, flags=re.DOTALL
    )

    # 2. Match all variations of tool calling syntax:
    # (a) <|tool_call|>call:name(args)<|tool_call|>
    # (b) <|im_start|>call:name(args)<|im_end|>
    # (c) <tool_code>name(args)</tool_code>
    # (d) <tool_call>name(args)</tool_call>
    tool_matches = []

    # Pattern 1: standard/custom hermes/gemma tool call
    p1 = re.finditer(
        r"<\|tool_call\|>call:([\w\.:]+)([\(\{].*?[\)\}])<\|tool_call\|>", content_str, re.DOTALL
    )
    for m in p1:
        tool_matches.append((m.group(0), m.group(1), m.group(2)))

    # Pattern 2: ChatML tool call
    p2 = re.finditer(r"<\|im_start\|>call:([\w\.:]+)([\(\{].*?[\)\}])", content_str, re.DOTALL)
    for m in p2:
        # We also want to capture a trailing <|im_end|> or <|im_end|>\n if present in the full match
        full_match = m.group(0)
        # Check if <|im_end|> immediately follows (with optional whitespace)
        rest = content_str[m.end() :]
        end_match = re.match(r"\s*<\|im_end\|>?", rest)
        if end_match:
            full_match += end_match.group(0)
        tool_matches.append((full_match, m.group(1), m.group(2)))

    # Pattern 3: xml tool tags
    p3 = re.finditer(
        r"<(?:tool_code|tool_call|function)>\s*([\w\.:]+)([\(\{].*?[\)\}])\s*</(?:tool_code|tool_call|function)>",
        content_str,
        re.DOTALL,
    )
    for m in p3:
        tool_matches.append((m.group(0), m.group(1), m.group(2)))

    synthesized_tool_calls = []
    for full_match, full_tool_name, args_raw in tool_matches:
        args_raw = args_raw.strip()
        if args_raw.startswith("(") and args_raw.endswith(")"):
            args_str = args_raw[1:-1]
        elif args_raw.startswith("{") and args_raw.endswith("}"):
            args_str = args_raw
        else:
            args_str = args_raw

        args_dict = parse_args_str(args_str)
        tool_name = full_tool_name.split(".")[-1].split(":")[-1]

        # Parameter mapping
        if "file_path" in args_dict and "path" not in args_dict:
            args_dict["path"] = args_dict["file_path"]
        if "filename" in args_dict and "path" not in args_dict:
            args_dict["path"] = args_dict["filename"]

        synthesized_tool_calls.append({"name": tool_name, "arguments": args_dict})

        content_str = content_str.replace(full_match, "").strip()

    # 3. Clean up known leaked assistant header and trailing im_end AFTER extracting tools and responses
    content_str = re.sub(r"<\|im_start\|>assistant\n?", "", content_str)
    content_str = re.sub(r"<\|im_end\|>?", "", content_str)
    content_str = re.sub(r"<\|thought\n?", "", content_str)

    # 4. Clean up any thinking tags or channels if they leaked
    content_str = re.sub(
        r"<\|channel>thought.*?(?:<channel\|>|<\|channel\|>|<channel>|$)",
        "",
        content_str,
        flags=re.DOTALL,
    )
    content_str = re.sub(r"<think>.*?</think>", "", content_str, flags=re.DOTALL)

    return synthesized_tool_calls, content_str.strip().strip()


class TestRegexToolParsing(unittest.TestCase):
    def test_regex_parsing_of_hermes_tool_calls(self):
        # Generic mock response content
        session_content = (
            "<|thought\n"
            "<|im_start|>call:write_file(path='/path/to/LlamaStudio/hello.txt', content='Hello from local Hermes CLI!')\n"
            "<|im_end|>\n"
            "<|im_start|>response:write_file(path='/path/to/LlamaStudio/hello.txt', content='Hello from local Hermes CLI!')\n"
            "{\n"
            '  "status": "success",\n'
            '  "message": "File written successfully",\n'
            '  "path": "/path/to/LlamaStudio/hello.txt"\n'
            "}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "File written: /path/to/LlamaStudio/hello.txt\n"
            "<|im_end|>"
        )

        synthesized_calls, cleaned_content = process_message_content(session_content)

        # We assert that tool calls are correctly extracted
        self.assertEqual(len(synthesized_calls), 1)
        self.assertEqual(synthesized_calls[0]["name"], "write_file")
        self.assertEqual(
            synthesized_calls[0]["arguments"]["path"], "/path/to/LlamaStudio/hello.txt"
        )
        self.assertEqual(
            synthesized_calls[0]["arguments"]["content"], "Hello from local Hermes CLI!"
        )

        # We assert that the leaked assistant tags and thoughts are removed from the clean display text
        self.assertEqual(cleaned_content, "File written: /path/to/LlamaStudio/hello.txt")


if __name__ == "__main__":
    unittest.main()
