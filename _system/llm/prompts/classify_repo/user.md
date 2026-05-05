<task>
You are given the README (and optional metadata) of a code repository plus the current state of our research taxonomy. Perform the following three tasks:

1. Picking a domain by index into the `existing_taxonomy` tree below. Each top-level node is a domain (prefixed with its index, e.g. `0.`, `1.`); its children are existing collections under that domain (prefixed with their own index, e.g. `0:`, `1:`). If no existing domain is a good fit, set `domain_index` to -1, provide a new domain name in `new_domain`, and provide a one-sentence, brief description of the domain (what it covers as a research area — not what this specific repository is about) in `new_domain_desc`. This description is stored with the domain and shown to you on future repos/papers, so it should describe the *area*, not the *repository*.
2. Picking a collection by index within the chosen domain. Set `collection_index` to the integer prefix of an existing collection (e.g. `0`, `1`) when one fits reasonably well. To propose a new collection, set `collection_index` to -1, put the new name in `new_collection`, and provide a one-sentence description of the collection (what family of work it groups — not what this specific repository is about) in `new_collection_desc`. This description is stored with the collection and shown to you on future repos/papers, so it should describe the *cluster*, not the *repository*.
3. Deeply understand the README (and metadata, if provided) and generate a set of semantic topics the repository covers. Topics are not the same thing as entities -- they capture an important semantic unit the repository discusses. Topics should be distinct from the chosen/created domain and collection. Topics are distinct from structural metadata like document titles, version identifiers, license tags, badge labels, install / usage commands, file paths, or CI status text. Favor the project's *research subject and approach* over surface README chrome. There is not a certain number of them you need to hit; the right number is whatever the source supports. Including a weakly-grounded topic to round out the list degrades the index, so when a candidate is borderline, leave it out. Prefer a short list of well-grounded topics over a longer list with weak entries. Here are a few ways you might validate generated topics:
  a. **Linking**: Would I semantically link two repositories if both discuss this? (If NO → don't extract)
  b. **Reusability**: Could this concept appear in other repositories in this domain? (If NO → too repo-specific)
  c. **Specificity**: Is this specific enough to be meaningful, but general enough to be reusable? (If NO → adjust)


Only use the README (and metadata, if provided) shown below — do not infer content that isn't present.
</task>

<existing_taxonomy>
{EXISTING_TAXONOMY}
</existing_taxonomy>

<readme>
{README_CONTENT}
</readme>

{METADATA_BLOCK}

<instructions>
1. Set `domain_index` to the integer index of the best-fitting existing domain, or -1 to propose a new one.
2. If `domain_index == -1`, set `new_domain` (the domain name) and `new_domain_desc` (a one-sentence description of the *research area*, not this repository). Otherwise leave both as empty strings.
3. Set `collection_index` to the integer index of an existing collection within the chosen domain, or -1 to propose a new collection.
4. If `collection_index == -1`, set `new_collection` (the collection name) and `new_collection_desc` (a one-sentence description of the *cluster of work*, not this repository). Otherwise leave both as empty strings.
5. If `domain_index == -1`, `collection_index` must be -1 — new domains have no existing collections.
6. Set `topics` to a list naming the repository's salient semantic concepts. Each topic must be distinct from entities, from the chosen domain/collection, and from structural metadata (badges, install commands, license tags, etc.), and must pass the linking/reusability/specificity tests in task 3. Let the count follow the content — emit only strongly-grounded topics, and do not pad to hit a target number.
</instructions>
