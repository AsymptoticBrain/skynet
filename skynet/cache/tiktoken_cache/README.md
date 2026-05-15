# Pre-bundled tiktoken encoding cache

Production runs (and Docker builds) are firewalled and cannot reach
`openaipublic.blob.core.windows.net`, which is where `tiktoken` would
normally download its encoding `.tiktoken` files on first use. To avoid the
SSL/network failures that result, the encoding files are committed to this
directory and `TIKTOKEN_CACHE_DIR` is pointed here at process start
(`skynet/env.py` for local dev, `ENV TIKTOKEN_CACHE_DIR=/app/tiktoken_cache`
in the runtime image).

## Expected files

`tiktoken` names cache entries by the SHA1 of the upstream blob URL, not by
the human encoding name. The files below must be present (download the
upstream URL on a machine with internet access and save the body under the
SHA1-named filename):

| Encoding       | Used by                              | Filename (SHA1)                            | Source URL |
|----------------|--------------------------------------|--------------------------------------------|------------|
| `o200k_base`   | gpt-4o family                        | `fb374d419588a4632f3f557e76b4b70aebbca790` | https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken |
| `cl100k_base`  | gpt-4, gpt-3.5-turbo, ada-002        | `9b5ad71b2ce5302211f9c61530b329a4922fc6a4` | https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken |
| `p50k_base`    | codex, text-davinci-002/003          | `ec7223a39ce59f226a68acc30dc1af2788490e15` | https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken |
| `r50k_base`    | gpt2 (langchain default splitter)    | `0ea1e91bbb3a60f729a8dc8f777fd2fc07cd8df4` | https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken |

## Populating

Run `scripts/warm_tiktoken_cache.py` on a machine with outbound internet
access; it writes the four files into this directory under their SHA1 names.
Then commit them.
