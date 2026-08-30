# Capsule Networks: architecture, routing, and the latent-space intuition

This document explains the model family that motivated this repository. It is written from the perspective of revisiting the architecture after the rise of Transformers and modern representation learning, while keeping the original 2017 formulation distinct from later interpretations.

## 1. What a capsule was trying to be

Sabour, Frosst, and Hinton describe a capsule as a group of neurons whose activity vector represents the **instantiation parameters of a particular kind of entity**, such as an object or object part. The vector's length is interpreted as evidence that the entity exists; the vector's orientation carries learned properties of the entity.

This is more structured than the usual scalar activation story.

A conventional scalar feature can roughly say:

> "this feature is present strongly."

A capsule vector is intended to say something closer to:

> "this kind of entity is present, and here is a learned coordinate describing its state."

The seductive part is that the state is not required to be named in advance. A capsule can learn a latent coordinate system whose directions jointly encode pose, style, deformation, local geometry, or whatever other properties are useful.

### Important correction to the common intuition

It is tempting to describe every capsule as a clean little semantic latent space where coordinate 0 is rotation, coordinate 1 is thickness, coordinate 2 is translation, and so on. The architecture does **not** guarantee that kind of disentanglement.

A learned representation can rotate, mix, and distribute factors however optimization finds useful. A more careful statement is:

> Each capsule is a vector-valued latent representation associated with an entity/feature type, and its geometry is intended to retain information that a scalar activation would discard.

That remains an interesting idea even if the basis is not human-readable.

## 2. Why CNN pooling bothered Hinton

The historical motivation was partly a criticism of how convolutional networks obtain invariance.

A CNN is very good at learning that a feature may occur in many positions. Pooling and related mechanisms help make higher layers less sensitive to exact location. But Hinton's complaint was that **throwing away pose information to obtain invariance is different from representing pose explicitly and transforming it predictably**.

Capsules aimed at **equivariance** rather than merely invariance:

- if an object moves or rotates, the representation should change in a structured way;
- the model should retain relationships among parts;
- higher-level objects should be supported when lower-level parts agree on a compatible whole.

This motivates the part-whole hierarchy.

## 3. The original vector-capsule pipeline

The Sabour et al. MNIST architecture can be summarized as:

```text
pixels
  -> ordinary convolutional features
  -> PrimaryCaps: groups of scalar feature maps become vectors
  -> DigitCaps: each lower capsule predicts each possible class capsule
  -> dynamic routing by agreement
  -> vector length selects the class
  -> optional reconstruction network regularizes the representation
```

The implementation in this repository follows that general shape, with a hand-written sequential reference and a CUDA representation.

### Lower-level capsules

Let a lower-level capsule output be:

\[
\mathbf{u}_i \in \mathbb{R}^{d_{in}}.
\]

Rather than sending one scalar to every parent, capsule \(i\) creates a different **vote** for each potential parent \(j\):

\[
\hat{\mathbf{u}}_{j|i} = \mathbf{W}_{ij}\mathbf{u}_i,
\]

where

\[
\mathbf{W}_{ij} \in \mathbb{R}^{d_{out}\times d_{in}}.
\]

That matrix is important. It says: "if I am this lower-level entity in this state, what state should parent type \(j\) have if I belong to it?"

The model therefore learns **relationships between entity types**, not only activations.

## 4. Dynamic routing by agreement

Routing starts with logits \(b_{ij}\), conventionally initialized to zero. They become normalized coupling coefficients:

\[
c_{ij} = \operatorname{softmax}_j(b_{ij}).
\]

A candidate parent receives a weighted sum of votes:

\[
\mathbf{s}_j = \sum_i c_{ij}\hat{\mathbf{u}}_{j|i}.
\]

The parent's output is the capsule squash function:

\[
\mathbf{v}_j = \operatorname{squash}(\mathbf{s}_j)
= \frac{\|\mathbf{s}_j\|^2}{1+\|\mathbf{s}_j\|^2}
  \frac{\mathbf{s}_j}{\|\mathbf{s}_j\|}.
\]

Then agreement updates routing logits:

\[
b_{ij} \leftarrow b_{ij} + \hat{\mathbf{u}}_{j|i}\cdot\mathbf{v}_j.
\]

The loop repeats a small fixed number of times.

The intuition is almost conversational:

1. every lower capsule predicts what each parent would look like;
2. tentative parents form from weighted votes;
3. each lower capsule asks which parent resembles its prediction;
4. agreement strengthens that route;
5. a new weighted parent is formed;
6. repeat.

The routing coefficients are therefore **input-dependent intermediate assignments**, not static learned weights.

That was one of the architecture's most compelling ideas.

## 5. A tiny inference process inside a forward pass

Dynamic routing can be understood as a small iterative inference procedure.

Most feed-forward layers apply learned parameters once and continue. Routing spends extra computation on the current example to infer which lower-level entities belong with which higher-level entities.

That makes the architecture feel less alien in an era where researchers routinely discuss:

- inference-time compute;
- iterative refinement;
- sparse routing and mixture-of-experts assignment;
- latent-variable inference;
- object-centric slot assignment.

The original routing mechanism is not equivalent to any of those techniques, but they share a willingness to let **computation on the current example determine information flow**.

## 6. Capsule vectors and LLM latent representations

A useful conceptual bridge is that LLMs do not manipulate English words internally in the ordinary linguistic sense. Tokens are mapped into vectors, and layers transform/contextualize those vectors through learned linear and nonlinear operations.

Likewise, a Capsule Network does not need a hand-authored vocabulary of "edge", "corner", "wheel", or "pose". It manipulates learned vectors whose geometry is determined by training.

A rough comparison is:

```text
CapsNet
pixels -> scalar features -> entity-like vectors -> routed relationships -> higher entity vectors

Transformer
symbols/patches -> embeddings -> contextual vectors -> attention-mediated relationships -> contextual vectors
```

The important difference is architectural commitment.

A Capsule Network imposes a stronger hypothesis:

> vector bundles correspond to entity-like representations, and lower-level entities should make transformation-based predictions of higher-level entities.

A Transformer imposes a weaker one:

> representations should be allowed to exchange information according to learned compatibility.

The weaker assumption proved far easier to scale.

## 7. Dynamic routing and attention: the analogy

Scaled dot-product attention is:

\[
\operatorname{Attention}(Q,K,V)
= \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V.
\]

At a high level, both mechanisms contain:

1. vector compatibility;
2. normalization into routing/attention weights;
3. weighted aggregation of vectors;
4. input-dependent information flow.

A loose mapping is:

| Dynamic routing | Attention-like interpretation |
|---|---|
| lower capsule \(u_i\) | source representation/token |
| transformed vote \(\hat u_{j|i}\) | key/value-like prediction conditioned on a possible destination |
| candidate parent \(v_j\) | query/prototype-like destination representation |
| \(\hat u_{j|i}\cdot v_j\) | compatibility score analogous to a query-key dot product |
| \(c_{ij}\) | normalized routing/attention weight |
| \(\sum_i c_{ij}\hat u_{j|i}\) | weighted value aggregation |

This is a **family resemblance, not an identity**.

### Where the analogy breaks

Dynamic routing does not cleanly factor each representation into independent learned Q, K, and V projections. The transformed vote participates in prediction and aggregation, while the parent representation itself emerges from the iterative routing process.

Attention usually computes compatibility and aggregation in one feed-forward operator. Original routing:

- initializes assignments;
- forms parents;
- compares votes with those parents;
- changes assignments;
- forms parents again.

So routing is closer to iterative clustering/assignment than one ordinary attention layer.

Later Capsule Network literature explicitly studies the relationship between routing and attention, and Efficient-CapsNet replaces iterative routing with a non-iterative self-attention routing mechanism. This is a useful historical clue: the conceptual bridge is real enough that researchers have operationalized it, while the exact algorithms remain different.

## 8. The margin loss

The original DigitCaps output uses vector length as class evidence. For class \(k\), the margin loss is:

\[
L_k = T_k\max(0,m^+ - \|v_k\|)^2
+ \lambda(1-T_k)\max(0,\|v_k\|-m^-)^2.
\]

Typical original values were approximately:

- \(m^+=0.9\)
- \(m^-=0.1\)
- \(\lambda=0.5\)

This repository's evolutionary work treats those values, plus capsule dimensions, tensor-channel count, and batch size, as search variables rather than sacred constants.

## 9. Reconstruction regularization

The original network masks all class capsules except the target and feeds the remaining vector into a decoder that reconstructs the image. The reconstruction objective encourages the active capsule to retain information about the input beyond the class label.

The sequential code in this repository contains reconstruction machinery, but portions of the revival branch's training path are commented/experimental. Reconstruction therefore needs to be treated as a historical feature requiring path reconstruction, not assumed active in every reported experiment.

## 10. Why the architecture was compelling

Capsules promised several things simultaneously:

- retain pose/instantiation information instead of discarding it;
- model part-whole relationships explicitly;
- learn transformation relationships among entity types;
- dynamically decide which higher entity should receive a lower entity's information;
- use richer vector state than scalar feature presence;
- potentially generalize better across viewpoint changes and overlapping objects.

This is more ambitious than "another classifier." It is an architectural theory of representation.

That ambition is also part of why the design was difficult to scale.

## 11. The systems consequence

For every lower capsule \(i\) and candidate parent \(j\), the vanilla model may need a transformation matrix and vote. With \(N\) lower capsules and \(M\) upper capsules, the relationship graph is potentially dense:

\[
N\times M.
\]

The transformation work is roughly proportional to:

\[
O(NM d_{in}d_{out}),
\]

before accounting for repeated routing, reductions, and synchronization.

That relationship graph is exactly what the CUDA thesis work attacks. The repository flattens the otherwise object-heavy capsule structures so this work can be mapped explicitly onto GPU grid/block dimensions.

For the implementation details, continue to [`CUDA_ARCHITECTURE.md`](CUDA_ARCHITECTURE.md).

## References

- Sabour, Frosst, Hinton. **Dynamic Routing Between Capsules** (2017): https://arxiv.org/abs/1710.09829
- Hinton, Sabour, Frosst. **Matrix Capsules with EM Routing** (ICLR 2018): https://openreview.net/forum?id=HJWLfGWRb
- Vaswani et al. **Attention Is All You Need** (2017): https://arxiv.org/abs/1706.03762
- Ribeiro et al. **Learning with Capsules: A Survey** (2022): https://arxiv.org/abs/2206.02664
- Mazzia, Salvetti, Chiaberge. **Efficient-CapsNet: capsule network with self-attention routing** (2021): https://pmc.ncbi.nlm.nih.gov/articles/PMC8290018/
