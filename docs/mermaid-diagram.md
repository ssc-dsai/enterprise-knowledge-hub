# EKH Sequence diagram

Here is the sequence diagram of our EKH service.


```mermaid
---
title: Enterprise Knowledge Hub Service flow
---
sequenceDiagram
    actor u as User
    participant API@{ "type": "boundary" } as EKH API
    u-->>API: Start run (Wikipedia)

    box Knowledge Service
    participant ingest as Ingesting
    participant process as Processing
    participant store as Storing
    participant impl as Wikipedia
    end

    participant queue@{ "type" : "queue" } as Queue

    participant db@{ "type": "database" } as Database

    API->>ingest: ingest()
    activate ingest
    ingest->>impl: fetch_from_source()
    Note left of impl: Read source material
    impl->>ingest: return KnowledgeItem(s)
    ingest->>queue: write KnowledgeItem(s)
    deactivate ingest

    API->>process: process()
    activate process
    process->>impl: process_item()
    impl->>process: return WikipediaItemProcessed
    Note left of impl: GPU processing of<br/>material
    process->>queue: emit_processed_item()
    deactivate process

    API->>store: store()
    activate store
    store->>queue: read WikipediaItemProcessed
    store->>db: store_item()
    deactivate store
 