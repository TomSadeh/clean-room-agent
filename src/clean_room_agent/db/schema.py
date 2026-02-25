"""DDL for all three databases — curated, raw, and session."""

import sqlite3


def create_curated_schema(conn: sqlite3.Connection) -> None:
    """Create curated DB tables and indexes. Idempotent."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS repos (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL,
            remote_url TEXT,
            indexed_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY,
            repo_id INTEGER NOT NULL REFERENCES repos(id),
            path TEXT NOT NULL,
            language TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            file_source TEXT NOT NULL DEFAULT 'project'
        );

        CREATE TABLE IF NOT EXISTS symbols (
            id INTEGER PRIMARY KEY,
            file_id INTEGER NOT NULL REFERENCES files(id),
            name TEXT NOT NULL,
            kind TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            signature TEXT,
            parent_symbol_id INTEGER REFERENCES symbols(id)
        );

        CREATE TABLE IF NOT EXISTS docstrings (
            id INTEGER PRIMARY KEY,
            symbol_id INTEGER REFERENCES symbols(id),
            file_id INTEGER NOT NULL REFERENCES files(id),
            content TEXT NOT NULL,
            format TEXT,
            parsed_fields TEXT
        );

        CREATE TABLE IF NOT EXISTS inline_comments (
            id INTEGER PRIMARY KEY,
            file_id INTEGER NOT NULL REFERENCES files(id),
            symbol_id INTEGER REFERENCES symbols(id),
            line INTEGER NOT NULL,
            content TEXT NOT NULL,
            kind TEXT,
            is_rationale INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS dependencies (
            id INTEGER PRIMARY KEY,
            source_file_id INTEGER NOT NULL REFERENCES files(id),
            target_file_id INTEGER NOT NULL REFERENCES files(id),
            kind TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS symbol_references (
            id INTEGER PRIMARY KEY,
            caller_symbol_id INTEGER NOT NULL REFERENCES symbols(id),
            callee_symbol_id INTEGER NOT NULL REFERENCES symbols(id),
            reference_kind TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS commits (
            id INTEGER PRIMARY KEY,
            repo_id INTEGER NOT NULL REFERENCES repos(id),
            hash TEXT NOT NULL,
            author TEXT,
            message TEXT,
            timestamp TEXT NOT NULL,
            files_changed INTEGER,
            insertions INTEGER,
            deletions INTEGER
        );

        CREATE TABLE IF NOT EXISTS file_commits (
            file_id INTEGER NOT NULL REFERENCES files(id),
            commit_id INTEGER NOT NULL REFERENCES commits(id),
            PRIMARY KEY (file_id, commit_id)
        );

        CREATE TABLE IF NOT EXISTS co_changes (
            file_a_id INTEGER NOT NULL REFERENCES files(id),
            file_b_id INTEGER NOT NULL REFERENCES files(id),
            count INTEGER NOT NULL DEFAULT 1,
            last_commit_hash TEXT,
            PRIMARY KEY (file_a_id, file_b_id),
            CHECK (file_a_id < file_b_id)
        );

        CREATE TABLE IF NOT EXISTS file_metadata (
            file_id INTEGER PRIMARY KEY REFERENCES files(id),
            purpose TEXT,
            module TEXT,
            domain TEXT,
            concepts TEXT,
            public_api_surface TEXT,
            complexity_notes TEXT
        );

        CREATE TABLE IF NOT EXISTS adapter_metadata (
            id INTEGER PRIMARY KEY,
            stage_name TEXT NOT NULL,
            base_model TEXT NOT NULL,
            model_tag TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            active INTEGER NOT NULL DEFAULT 1,
            performance_notes TEXT,
            deployed_at TEXT NOT NULL
        );

        -- Indexes
        CREATE UNIQUE INDEX IF NOT EXISTS idx_repos_path ON repos(path);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_files_repo_path ON files(repo_id, path);
        CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_id);
        CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
        CREATE INDEX IF NOT EXISTS idx_symbol_refs_caller ON symbol_references(caller_symbol_id);
        CREATE INDEX IF NOT EXISTS idx_symbol_refs_callee ON symbol_references(callee_symbol_id);
        CREATE INDEX IF NOT EXISTS idx_deps_source ON dependencies(source_file_id);
        CREATE INDEX IF NOT EXISTS idx_deps_target ON dependencies(target_file_id);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_commits_repo_hash ON commits(repo_id, hash);
        CREATE INDEX IF NOT EXISTS idx_file_metadata_domain ON file_metadata(domain);
        CREATE INDEX IF NOT EXISTS idx_file_metadata_module ON file_metadata(module);
        CREATE INDEX IF NOT EXISTS idx_adapter_stage_active ON adapter_metadata(stage_name, active);
    """)

    # Migration: add file_source column for existing DBs
    try:
        conn.execute("ALTER TABLE files ADD COLUMN file_source TEXT NOT NULL DEFAULT 'project'")
    except sqlite3.OperationalError:
        pass  # Column already exists


def create_raw_schema(conn: sqlite3.Connection) -> None:
    """Create raw DB tables. Idempotent."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS index_runs (
            id INTEGER PRIMARY KEY,
            repo_path TEXT NOT NULL,
            files_scanned INTEGER NOT NULL,
            files_changed INTEGER NOT NULL,
            duration_ms INTEGER NOT NULL,
            status TEXT NOT NULL,
            timestamp TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS enrichment_outputs (
            id INTEGER PRIMARY KEY,
            task_id TEXT,
            file_id INTEGER NOT NULL,
            file_path TEXT,
            model TEXT NOT NULL,
            purpose TEXT,
            module TEXT,
            domain TEXT,
            concepts TEXT,
            public_api_surface TEXT,
            complexity_notes TEXT,
            system_prompt TEXT,
            raw_prompt TEXT NOT NULL,
            raw_response TEXT NOT NULL,
            promoted INTEGER NOT NULL DEFAULT 0,
            timestamp TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS retrieval_llm_calls (
            id INTEGER PRIMARY KEY,
            task_id TEXT NOT NULL,
            call_type TEXT NOT NULL,
            stage_name TEXT,
            model TEXT NOT NULL,
            system_prompt TEXT,
            prompt TEXT NOT NULL,
            response TEXT NOT NULL,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            latency_ms INTEGER NOT NULL,
            timestamp TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS retrieval_decisions (
            id INTEGER PRIMARY KEY,
            task_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            file_id INTEGER NOT NULL,
            symbol_id INTEGER,
            detail_level TEXT,
            tier TEXT,
            included INTEGER NOT NULL,
            reason TEXT,
            timestamp TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS task_runs (
            id INTEGER PRIMARY KEY,
            task_id TEXT NOT NULL UNIQUE,
            repo_path TEXT NOT NULL,
            mode TEXT NOT NULL,
            execute_model TEXT NOT NULL,
            context_window INTEGER NOT NULL,
            reserved_tokens INTEGER NOT NULL,
            stages TEXT NOT NULL,
            plan_artifact TEXT,
            success INTEGER,
            total_tokens INTEGER,
            total_latency_ms INTEGER,
            final_diff TEXT,
            final_plan TEXT,
            timestamp TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS run_attempts (
            id INTEGER PRIMARY KEY,
            task_run_id INTEGER NOT NULL REFERENCES task_runs(id),
            attempt INTEGER NOT NULL,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            latency_ms INTEGER NOT NULL,
            raw_response TEXT NOT NULL,
            patch_applied INTEGER NOT NULL,
            timestamp TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS validation_results (
            id INTEGER PRIMARY KEY,
            attempt_id INTEGER NOT NULL REFERENCES run_attempts(id),
            success INTEGER NOT NULL,
            test_output TEXT,
            lint_output TEXT,
            type_check_output TEXT,
            failing_tests TEXT
        );

        CREATE TABLE IF NOT EXISTS session_archives (
            id INTEGER PRIMARY KEY,
            task_id TEXT NOT NULL,
            session_blob BLOB NOT NULL,
            archived_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS training_plans (
            id INTEGER PRIMARY KEY,
            task_id TEXT NOT NULL,
            target_stage TEXT NOT NULL,
            base_model TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'logged',
            data_criteria TEXT NOT NULL,
            hyperparameters TEXT,
            improvement_targets TEXT,
            raw_plan TEXT NOT NULL,
            timestamp TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS training_datasets (
            id INTEGER PRIMARY KEY,
            training_plan_id INTEGER NOT NULL REFERENCES training_plans(id),
            stage_name TEXT NOT NULL,
            base_model TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'logged',
            dataset_path TEXT NOT NULL,
            example_count INTEGER NOT NULL,
            positive_count INTEGER NOT NULL,
            negative_count INTEGER NOT NULL,
            format TEXT NOT NULL DEFAULT 'jsonl',
            timestamp TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS adapter_registry (
            id INTEGER PRIMARY KEY,
            dataset_id INTEGER NOT NULL REFERENCES training_datasets(id),
            stage_name TEXT NOT NULL,
            base_model TEXT NOT NULL,
            model_tag TEXT NOT NULL,
            training_loss REAL,
            eval_metrics TEXT,
            deployed INTEGER NOT NULL DEFAULT 0,
            trained_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS orchestrator_runs (
            id INTEGER PRIMARY KEY,
            task_id TEXT NOT NULL,
            repo_path TEXT NOT NULL,
            task_description TEXT NOT NULL,
            total_parts INTEGER,
            total_steps INTEGER,
            parts_completed INTEGER NOT NULL DEFAULT 0,
            steps_completed INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            completed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS orchestrator_passes (
            id INTEGER PRIMARY KEY,
            orchestrator_run_id INTEGER NOT NULL REFERENCES orchestrator_runs(id),
            task_run_id INTEGER NOT NULL REFERENCES task_runs(id),
            pass_type TEXT NOT NULL,
            part_id TEXT,
            step_id TEXT,
            sequence_order INTEGER NOT NULL,
            timestamp TEXT NOT NULL
        );
    """)

    # Migration: add file_path column for existing DBs (8.2 fix).
    # SQLite has no ALTER TABLE ADD COLUMN IF NOT EXISTS, so catch the
    # "duplicate column" error.  This is the one intentional silent-catch
    # in the codebase — the migration is idempotent by construction.
    try:
        conn.execute("ALTER TABLE enrichment_outputs ADD COLUMN file_path TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists (new DB or migration already applied)

    # Migration: add git fields to orchestrator_runs
    try:
        conn.execute("ALTER TABLE orchestrator_runs ADD COLUMN git_branch TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE orchestrator_runs ADD COLUMN git_base_ref TEXT")
    except sqlite3.OperationalError:
        pass


def create_session_schema(conn: sqlite3.Connection) -> None:
    """Create session DB table. Idempotent."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS kv (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    """)
