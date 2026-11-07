# Deviations

Each entry: what differs from the manuscript, the paper anchor, and why.

## D1 — Descriptor embeddings
- Anchor: §IV-C (PubMedBERT 110M frozen text encoder; BiomedCLIP-pretrained visual init).
- The model currently uses a deterministic descriptor-embedding table. A helper for loading
  PubMedBERT embeddings exists but is not connected to model initialization. BiomedCLIP visual
  initialization is not implemented.

## D4 — Token grid produced by a /16 patch embedding
- Anchor: §IV-C ("encoded down to H/16 x W/16 x D/16 ... N = 12 x 12 x 3 = 432 tokens ... patch 4x4x2").
- Deviation: the visual backbone reaches the reported 12 x 12 x 3 = 432 token grid via a single /16
  strided patch embedding followed by four constant-resolution shifted-window stages with the reported
  channel progression [96,192,384,768], rather than a patch(4x4x2)-embed plus three patch-merging
  steps (which would yield 6 x 6 x 3 = 108 tokens and break the reported action space 1297 = 3 x 432 + 1).
- Why: the action space |A| = 1297 and the complexity analysis depend on exactly 432 tokens per
  sequence and the stated /16 resolution; honouring the reported token count and channel widths is the
  binding constraint. Token count is computed at runtime from the encoder output.
