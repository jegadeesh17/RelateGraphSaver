# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.4.html).

## [Unreleased]

### Added
- Core `RelateSaver` checkpointer implementing `BaseCheckpointSaver` interface for LangGraph with thread-safe persistence.
- Embedded graph storage backed by Relate DB engine with Cypher DDL schema constraints and `:NEXT` relationship modeling.
- Dedicated `RelateThreadProxy` actor queue maintaining single-thread affinity for Relate DB connections.
- State serialization with LangGraph typed serialization and deterministic SHA-256 state hashing.
- Base exception taxonomy and error mapping for database, concurrency, and serialization failures.
- Write-ahead log (WAL) synchronization and crash recovery support for durable checkpointing.
- Integration test suite validating end-to-end checkpointing, pagination, history retrieval, and multi-threaded execution.
