<task>
You are given the abstract and introduction of an arxiv paper and the current state of our research taxonomy. Perform the following three tasks:

1. Picking a domain by index into the `existing_taxonomy` tree below. Each top-level node is a domain (prefixed with its index, e.g. `0.`, `1.`); its children are existing collections under that domain (prefixed with their own index, e.g. `0:`, `1:`). If no existing domain is a good fit, set `domain_index` to -1, provide a new domain name in `new_domain`, and provide a one-sentence, brief description of the domain (what it covers as a research area — not what this specific paper is about) in `new_domain_desc`. This description is stored with the domain and shown to you on future papers, so it should describe the *area*, not the *paper*.
2. Picking one or more collections within the chosen domain. Return a `collections` array with 1..4 entries. The FIRST entry (index 0) is the PRIMARY collection — the paper's home, where someone browsing for this work would expect to find it. Entries 1..3 are SECONDARY memberships: include a secondary only when the paper substantively contributes to that category (real overlap, not just "mentions it" or "compares against it"). Most papers belong to exactly ONE collection — emit one entry then. Each entry has its own `index`/`new_name`/`new_desc` triple: set `index` to the integer prefix of an existing collection (e.g. `0`, `1`) when one fits, or to -1 to propose a new collection (then fill `new_name` and `new_desc`). Existing-index entries must leave `new_name` and `new_desc` empty. The new-collection description should describe the *cluster of work*, not the *paper* — it's stored with the collection and shown to you on future papers.
3. Deeply understand the abstract and introduction and generate a set of semantic topics the paper covers. Topics are not the same thing as entities -- they capture an important semantic unit the paper discusses. Topics should be distinct from the chosen/created domain and collections. Topics are distinct from structural metadata like document titles, section numbers, dates, version identifiers etc. There is not a certain number of them you need to hit; the right number is whatever the source supports. Including a weakly-grounded topic to round out the list degrades the index, so when a candidate is borderline, leave it out. Prefer a short list of well-grounded topics over a longer list with weak entries. Here are a few ways you might validate generated topics:
  a. **Linking**: Would I semantically link two papers if both discuss this? (If NO → don't extract)
  b. **Reusability**: Could this concept appear in other documents in this domain? (If NO → too document-specific)
  c. **Specificity**: Is this specific enough to be meaningful, but general enough to be reusable? (If NO → adjust)

Only use the paper content shown below — do not infer content that isn't present.
</task>

<existing_taxonomy>
{EXISTING_TAXONOMY}
</existing_taxonomy>

<paper_content>
{PAPER_CONTENT}
</paper_content>

<instructions>
1. Set `domain_index` to the integer index of the best-fitting existing domain, or -1 to propose a new one.
2. If `domain_index == -1`, set `new_domain` (the domain name) and `new_domain_desc` (a one-sentence description of the *research area*, not this paper). Otherwise leave both as empty strings.
3. Set `collections` to a 1..4-entry array. Entry 0 is the PRIMARY collection (the paper's home). Entries 1..3 are SECONDARY memberships, included only when the paper substantively contributes to that category. Most papers should have exactly one entry. For each entry, set `index` to the integer prefix of an existing collection within the chosen domain, or -1 to propose a new collection.
4. For each entry where `index == -1`, fill `new_name` (the new collection name) and `new_desc` (a one-sentence description of the *cluster of work*, not this paper). For each entry where `index >= 0`, leave both as empty strings.
5. If `domain_index == -1`, every entry's `index` must be -1 — new domains have no existing collections.
6. Set `topics` to a list naming the paper's salient semantic concepts. Each topic must be distinct from entities, from the chosen domain/collections, and from structural metadata, and must pass the linking/reusability/specificity tests in task 3. Let the count follow the content — emit only strongly-grounded topics, and do not pad to hit a target number.
</instructions>
