<task>
You are given an arxiv paper and the current state of our research taxonomy. Classify the paper by:

1. Picking a domain by index into the `existing_taxonomy` tree below. Each top-level node is a domain (prefixed with its index, e.g. `0.`, `1.`); its children are existing collections under that domain (prefixed with their own index, e.g. `0:`, `1:`). If no existing domain is a good fit, set `domain_index` to -1, provide a new domain name in `proposed_new_domain`, and provide a one-sentence description of the domain (what it covers as a research area — not what this specific paper is about) in `proposed_new_domain_description`. This description is stored with the domain and shown to you on future papers, so it should describe the *area*, not the *paper*.
2. Picking a collection by index within the chosen domain. Set `collection_index` to the integer prefix of an existing collection (e.g. `0`, `1`) when one fits reasonably well. To propose a new collection, set `collection_index` to -1, put the new name in `proposed_new_collection`, and provide a one-sentence description of the collection (what family of work it groups — not what this specific paper is about) in `proposed_new_collection_description`. This description is stored with the collection and shown to you on future papers, so it should describe the *cluster*, not the *paper*.
3. Listing the paper's most salient topics as short strings. Topics are resolved against the existing vocabulary automatically, so use the most natural phrasing for each.

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
2. If `domain_index == -1`, set `proposed_new_domain` (the domain name) and `proposed_new_domain_description` (a one-sentence description of the *research area*, not this paper). Otherwise leave both as empty strings.
3. Set `collection_index` to the integer index of an existing collection within the chosen domain, or -1 to propose a new collection.
4. If `collection_index == -1`, set `proposed_new_collection` (the collection name) and `proposed_new_collection_description` (a one-sentence description of the *cluster of work*, not this paper). Otherwise leave both as empty strings.
5. If `domain_index == -1`, `collection_index` must be -1 — new domains have no existing collections.
6. Set `topics` to an array of short strings naming the paper's salient topics.
</instructions>
