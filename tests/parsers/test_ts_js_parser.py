"""Tests for parsers/ts_js_parser.py."""

from clean_room_agent.parsers.ts_js_parser import TSJSParser

SAMPLE_TS = b'''
/**
 * A utility module for greeting.
 * @module greeting
 */

import { User } from "./models";
import type { Config } from "./config";

// TODO: add i18n support
const MAX_RETRIES = 3;

interface Greeter {
    greet(name: string): string;
}

/**
 * A class that generates greetings.
 * @param prefix - The greeting prefix.
 */
class HelloGreeter implements Greeter {
    private prefix: string;

    constructor(prefix: string) {
        this.prefix = prefix;
    }

    greet(name: string): string {
        // This returns the full greeting because we need consistent formatting
        return `${this.prefix}, ${name}!`;
    }
}

type GreetingStyle = "formal" | "casual";

enum Priority {
    Low,
    High,
}

/**
 * Create a greeter with the given style.
 * @param style - The greeting style.
 * @returns A new Greeter instance.
 */
function createGreeter(style: GreetingStyle): Greeter {
    return new HelloGreeter(style === "formal" ? "Good day" : "Hey");
}

const greetAll = (names: string[]): string[] => {
    return names.map(n => `Hello, ${n}`);
};

export { HelloGreeter, createGreeter };
'''

SAMPLE_JS = b'''
const path = require("path");
const { readFile } = require("fs");

// FIXME: handle edge cases
const DEFAULT_TIMEOUT = 5000;

/**
 * Process a file at the given path.
 * @param {string} filePath - The file path.
 * @returns {Promise<string>} The file contents.
 */
function processFile(filePath) {
    return readFile(filePath, "utf-8");
}

class FileProcessor {
    constructor(basePath) {
        this.basePath = basePath;
    }

    process(name) {
        return processFile(path.join(this.basePath, name));
    }
}

module.exports = { processFile, FileProcessor };
'''


class TestTypeScriptParser:
    def setup_method(self):
        self.parser = TSJSParser("typescript")

    def test_symbol_extraction(self):
        result = self.parser.parse(SAMPLE_TS, "test.ts")
        names = {s.name for s in result.symbols}
        assert "HelloGreeter" in names
        assert "createGreeter" in names
        assert "greetAll" in names
        assert "MAX_RETRIES" in names

    def test_interface_and_type(self):
        result = self.parser.parse(SAMPLE_TS, "test.ts")
        kinds = {s.name: s.kind for s in result.symbols}
        assert kinds.get("Greeter") == "interface"
        assert kinds.get("GreetingStyle") == "type_alias"
        assert kinds.get("Priority") == "enum"

    def test_class_methods(self):
        result = self.parser.parse(SAMPLE_TS, "test.ts")
        methods = [s for s in result.symbols if s.parent_name == "HelloGreeter"]
        method_names = {m.name for m in methods}
        assert "constructor" in method_names or "greet" in method_names

    def test_docstring_extraction(self):
        result = self.parser.parse(SAMPLE_TS, "test.ts")
        assert len(result.docstrings) > 0
        jsdocs = [d for d in result.docstrings if d.format == "jsdoc"]
        assert len(jsdocs) >= 1

    def test_comment_classification(self):
        result = self.parser.parse(SAMPLE_TS, "test.ts")
        todos = [c for c in result.comments if c.kind == "todo"]
        assert len(todos) >= 1
        rationale = [c for c in result.comments if c.is_rationale]
        assert len(rationale) >= 1

    def test_import_extraction(self):
        result = self.parser.parse(SAMPLE_TS, "test.ts")
        modules = {i.module for i in result.imports}
        assert "./models" in modules
        assert "./config" in modules
        # Check type-only import
        type_imports = [i for i in result.imports if i.is_type_only]
        assert len(type_imports) >= 1

    def test_no_symbol_references(self):
        result = self.parser.parse(SAMPLE_TS, "test.ts")
        assert result.references == []

    def test_tsx_file(self):
        tsx_source = b'function App(): JSX.Element { return <div>Hello</div>; }'
        result = self.parser.parse(tsx_source, "test.tsx")
        names = {s.name for s in result.symbols}
        assert "App" in names

    def test_arrow_function(self):
        result = self.parser.parse(SAMPLE_TS, "test.ts")
        arrows = [s for s in result.symbols if s.name == "greetAll"]
        assert len(arrows) == 1
        assert arrows[0].kind == "function"


class TestJavaScriptParser:
    def setup_method(self):
        self.parser = TSJSParser("javascript")

    def test_symbol_extraction(self):
        result = self.parser.parse(SAMPLE_JS, "test.js")
        names = {s.name for s in result.symbols}
        assert "processFile" in names
        assert "FileProcessor" in names

    def test_commonjs_imports(self):
        result = self.parser.parse(SAMPLE_JS, "test.js")
        modules = {i.module for i in result.imports}
        assert "path" in modules

    def test_comment_classification(self):
        result = self.parser.parse(SAMPLE_JS, "test.js")
        fixmes = [c for c in result.comments if c.kind == "fixme"]
        assert len(fixmes) >= 1

    def test_docstring_extraction(self):
        result = self.parser.parse(SAMPLE_JS, "test.js")
        jsdocs = [d for d in result.docstrings if d.format == "jsdoc"]
        assert len(jsdocs) >= 1
