"""KnowledgeBase query API — reads exclusively from the curated DB."""

import sqlite3

from clean_room_agent.query.models import (
    AdapterInfo,
    CoChange,
    Comment,
    Commit,
    Dependency,
    Docstring,
    File,
    FileContext,
    FileMetadata,
    RepoOverview,
    Symbol,
)


class KnowledgeBase:
    """Query API for the curated knowledge base."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    # --- File queries ---

    def get_files(self, repo_id: int, language: str | None = None) -> list[File]:
        if language:
            rows = self._conn.execute(
                "SELECT * FROM files WHERE repo_id = ? AND language = ?",
                (repo_id, language),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM files WHERE repo_id = ?", (repo_id,)
            ).fetchall()
        return [self._row_to_file(r) for r in rows]

    def get_file_by_id(self, file_id: int) -> File | None:
        row = self._conn.execute(
            "SELECT * FROM files WHERE id = ?", (file_id,)
        ).fetchone()
        return self._row_to_file(row) if row else None

    def get_file_by_path(self, repo_id: int, path: str) -> File | None:
        row = self._conn.execute(
            "SELECT * FROM files WHERE repo_id = ? AND path = ?",
            (repo_id, path),
        ).fetchone()
        return self._row_to_file(row) if row else None

    def search_files_by_metadata(
        self,
        repo_id: int,
        domain: str | None = None,
        module: str | None = None,
        concepts: str | None = None,
    ) -> list[File]:
        """Search files by enrichment metadata. Returns empty if unpopulated."""
        if domain is None and module is None and concepts is None:
            raise ValueError(
                "search_files_by_metadata requires at least one filter (domain, module, or concepts)"
            )
        conditions = ["f.repo_id = ?"]
        params: list = [repo_id]

        if domain:
            conditions.append("m.domain = ?")
            params.append(domain)
        if module:
            conditions.append("m.module = ?")
            params.append(module)
        if concepts:
            escaped = concepts.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            conditions.append("m.concepts LIKE ? ESCAPE '\\'")
            params.append(f"%{escaped}%")

        query = (
            "SELECT f.* FROM files f "
            "JOIN file_metadata m ON f.id = m.file_id "
            f"WHERE {' AND '.join(conditions)}"
        )
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_file(r) for r in rows]

    def get_library_files(self, repo_id: int) -> list[File]:
        """Get all files with file_source='library' for a repo."""
        rows = self._conn.execute(
            "SELECT * FROM files WHERE repo_id = ? AND file_source = 'library'",
            (repo_id,),
        ).fetchall()
        return [self._row_to_file(r) for r in rows]

    def get_file_metadata(self, file_id: int) -> FileMetadata | None:
        """Get enrichment metadata for a single file."""
        row = self._conn.execute(
            "SELECT * FROM file_metadata WHERE file_id = ?", (file_id,)
        ).fetchone()
        if not row:
            return None
        return FileMetadata(
            file_id=row["file_id"],
            purpose=row["purpose"],
            module=row["module"],
            domain=row["domain"],
            concepts=row["concepts"],
            public_api_surface=row["public_api_surface"],
            complexity_notes=row["complexity_notes"],
        )

    def get_file_metadata_batch(self, file_ids: list[int]) -> dict[int, FileMetadata]:
        """Get enrichment metadata for multiple files. Returns {file_id: FileMetadata}."""
        if not file_ids:
            return {}
        placeholders = ",".join("?" * len(file_ids))
        rows = self._conn.execute(
            f"SELECT * FROM file_metadata WHERE file_id IN ({placeholders})",
            file_ids,
        ).fetchall()
        return {
            row["file_id"]: FileMetadata(
                file_id=row["file_id"],
                purpose=row["purpose"],
                module=row["module"],
                domain=row["domain"],
                concepts=row["concepts"],
                public_api_surface=row["public_api_surface"],
                complexity_notes=row["complexity_notes"],
            )
            for row in rows
        }

    # --- Symbol queries ---

    def get_symbol_by_id(self, symbol_id: int) -> Symbol | None:
        row = self._conn.execute(
            "SELECT * FROM symbols WHERE id = ?", (symbol_id,)
        ).fetchone()
        return self._row_to_symbol(row) if row else None

    def get_symbols_for_file(self, file_id: int, kind: str | None = None) -> list[Symbol]:
        if kind:
            rows = self._conn.execute(
                "SELECT * FROM symbols WHERE file_id = ? AND kind = ?",
                (file_id, kind),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM symbols WHERE file_id = ?", (file_id,)
            ).fetchall()
        return [self._row_to_symbol(r) for r in rows]

    def search_symbols_by_name(self, repo_id: int, pattern: str) -> list[Symbol]:
        escaped = pattern.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        rows = self._conn.execute(
            "SELECT s.* FROM symbols s "
            "JOIN files f ON s.file_id = f.id "
            "WHERE f.repo_id = ? AND s.name LIKE ? ESCAPE '\\'",
            (repo_id, f"%{escaped}%"),
        ).fetchall()
        return [self._row_to_symbol(r) for r in rows]

    def get_symbol_neighbors(
        self,
        symbol_id: int,
        direction: str,
        kinds: list[str] | None = None,
    ) -> list[Symbol]:
        """Get symbols connected via symbol_references. Python MVP only."""
        if direction == "callees":
            rows = self._conn.execute(
                "SELECT s.* FROM symbols s "
                "JOIN symbol_references r ON s.id = r.callee_symbol_id "
                "WHERE r.caller_symbol_id = ?",
                (symbol_id,),
            ).fetchall()
        elif direction == "callers":
            rows = self._conn.execute(
                "SELECT s.* FROM symbols s "
                "JOIN symbol_references r ON s.id = r.caller_symbol_id "
                "WHERE r.callee_symbol_id = ?",
                (symbol_id,),
            ).fetchall()
        else:
            raise ValueError(f"direction must be 'callers' or 'callees', got {direction!r}")

        symbols = [self._row_to_symbol(r) for r in rows]
        if kinds:
            symbols = [s for s in symbols if s.kind in kinds]
        return symbols

    # --- Dependency queries ---

    def get_dependencies(self, file_id: int, direction: str) -> list[Dependency]:
        if direction == "imports":
            rows = self._conn.execute(
                "SELECT * FROM dependencies WHERE source_file_id = ?", (file_id,)
            ).fetchall()
        elif direction == "imported_by":
            rows = self._conn.execute(
                "SELECT * FROM dependencies WHERE target_file_id = ?", (file_id,)
            ).fetchall()
        else:
            raise ValueError(f"direction must be 'imports' or 'imported_by', got {direction!r}")
        return [self._row_to_dep(r) for r in rows]

    def get_dependency_subgraph(self, file_ids: list[int], depth: int) -> list[Dependency]:
        """BFS expansion from seed files up to `depth` hops."""
        seen_files = set(file_ids)
        seen_dep_ids: set[int] = set()
        all_deps = []
        frontier = set(file_ids)

        for _ in range(depth):
            if not frontier:
                break
            placeholders = ",".join("?" * len(frontier))
            rows = self._conn.execute(
                f"SELECT * FROM dependencies WHERE source_file_id IN ({placeholders}) "
                f"OR target_file_id IN ({placeholders})",
                list(frontier) + list(frontier),
            ).fetchall()
            new_frontier = set()
            for r in rows:
                dep = self._row_to_dep(r)
                if dep.id in seen_dep_ids:
                    continue
                seen_dep_ids.add(dep.id)
                all_deps.append(dep)
                for fid in (dep.source_file_id, dep.target_file_id):
                    if fid not in seen_files:
                        seen_files.add(fid)
                        new_frontier.add(fid)
            frontier = new_frontier

        return all_deps

    # --- Co-change queries ---

    def get_co_change_neighbors(self, file_id: int, min_count: int) -> list[CoChange]:
        rows = self._conn.execute(
            "SELECT * FROM co_changes "
            "WHERE (file_a_id = ? OR file_b_id = ?) AND count >= ?",
            (file_id, file_id, min_count),
        ).fetchall()
        return [
            CoChange(
                file_a_id=r["file_a_id"],
                file_b_id=r["file_b_id"],
                count=r["count"],
                last_commit_hash=r["last_commit_hash"],
            )
            for r in rows
        ]

    # --- Docstring / comment queries ---

    def get_docstrings_for_file(self, file_id: int) -> list[Docstring]:
        rows = self._conn.execute(
            "SELECT * FROM docstrings WHERE file_id = ?", (file_id,)
        ).fetchall()
        return [self._row_to_docstring(r) for r in rows]

    def get_rationale_comments(self, file_id: int) -> list[Comment]:
        rows = self._conn.execute(
            "SELECT * FROM inline_comments WHERE file_id = ? AND is_rationale = 1",
            (file_id,),
        ).fetchall()
        return [self._row_to_comment(r) for r in rows]

    # --- Commit queries ---

    def get_recent_commits_for_file(self, file_id: int, limit: int = 10) -> list[Commit]:
        rows = self._conn.execute(
            "SELECT c.* FROM commits c "
            "JOIN file_commits fc ON c.id = fc.commit_id "
            "WHERE fc.file_id = ? ORDER BY c.timestamp DESC LIMIT ?",
            (file_id, limit),
        ).fetchall()
        return [self._row_to_commit(r) for r in rows]

    # --- Composite queries ---

    def get_file_context(self, file_id: int) -> FileContext | None:
        row = self._conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
        if not row:
            return None
        file = self._row_to_file(row)
        return FileContext(
            file=file,
            symbols=self.get_symbols_for_file(file_id),
            docstrings=self.get_docstrings_for_file(file_id),
            rationale_comments=self.get_rationale_comments(file_id),
            dependencies=self.get_dependencies(file_id, "imports"),
            co_changes=self.get_co_change_neighbors(file_id, min_count=2),
            recent_commits=self.get_recent_commits_for_file(file_id),
        )

    def get_repo_overview(self, repo_id: int) -> RepoOverview:
        # File count
        count = self._conn.execute(
            "SELECT COUNT(*) FROM files WHERE repo_id = ?", (repo_id,)
        ).fetchone()[0]

        # Language distribution
        lang_rows = self._conn.execute(
            "SELECT language, COUNT(*) as cnt FROM files WHERE repo_id = ? GROUP BY language",
            (repo_id,),
        ).fetchall()
        language_counts = {r["language"]: r["cnt"] for r in lang_rows}

        # Domain distribution (from enrichment, may be empty)
        domain_rows = self._conn.execute(
            "SELECT m.domain, COUNT(*) as cnt FROM file_metadata m "
            "JOIN files f ON m.file_id = f.id "
            "WHERE f.repo_id = ? AND m.domain IS NOT NULL GROUP BY m.domain",
            (repo_id,),
        ).fetchall()
        domain_counts = {r["domain"]: r["cnt"] for r in domain_rows}

        # Most connected files (by dependency count)
        dep_rows = self._conn.execute(
            "SELECT f.path, "
            "(SELECT COUNT(*) FROM dependencies WHERE source_file_id = f.id) + "
            "(SELECT COUNT(*) FROM dependencies WHERE target_file_id = f.id) as dep_count "
            "FROM files f WHERE f.repo_id = ? ORDER BY dep_count DESC LIMIT 10",
            (repo_id,),
        ).fetchall()
        most_connected = [(r["path"], r["dep_count"]) for r in dep_rows]

        return RepoOverview(
            repo_id=repo_id,
            file_count=count,
            language_counts=language_counts,
            domain_counts=domain_counts,
            most_connected_files=most_connected,
        )

    # --- Adapter queries ---

    def get_adapter_for_stage(self, stage_name: str) -> AdapterInfo | None:
        row = self._conn.execute(
            "SELECT * FROM adapter_metadata WHERE stage_name = ? AND active = 1 "
            "ORDER BY version DESC LIMIT 1",
            (stage_name,),
        ).fetchone()
        if not row:
            return None
        return AdapterInfo(
            id=row["id"],
            stage_name=row["stage_name"],
            base_model=row["base_model"],
            model_tag=row["model_tag"],
            version=row["version"],
            performance_notes=row["performance_notes"],
            deployed_at=row["deployed_at"],
        )

    # --- Row converters ---

    @staticmethod
    def _row_to_file(row) -> File:
        return File(
            id=row["id"],
            repo_id=row["repo_id"],
            path=row["path"],
            language=row["language"],
            content_hash=row["content_hash"],
            size_bytes=row["size_bytes"],
            file_source=row["file_source"],
        )

    @staticmethod
    def _row_to_symbol(row) -> Symbol:
        return Symbol(
            id=row["id"],
            file_id=row["file_id"],
            name=row["name"],
            kind=row["kind"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            signature=row["signature"],
            parent_symbol_id=row["parent_symbol_id"],
        )

    @staticmethod
    def _row_to_docstring(row) -> Docstring:
        return Docstring(
            id=row["id"],
            file_id=row["file_id"],
            content=row["content"],
            format=row["format"],
            parsed_fields=row["parsed_fields"],
            symbol_id=row["symbol_id"],
        )

    @staticmethod
    def _row_to_comment(row) -> Comment:
        return Comment(
            id=row["id"],
            file_id=row["file_id"],
            line=row["line"],
            content=row["content"],
            kind=row["kind"],
            is_rationale=bool(row["is_rationale"]),
            symbol_id=row["symbol_id"],
        )

    @staticmethod
    def _row_to_dep(row) -> Dependency:
        return Dependency(
            id=row["id"],
            source_file_id=row["source_file_id"],
            target_file_id=row["target_file_id"],
            kind=row["kind"],
        )

    @staticmethod
    def _row_to_commit(row) -> Commit:
        return Commit(
            id=row["id"],
            repo_id=row["repo_id"],
            hash=row["hash"],
            author=row["author"],
            message=row["message"],
            timestamp=row["timestamp"],
            files_changed=row["files_changed"],
            insertions=row["insertions"],
            deletions=row["deletions"],
        )
