# Why Capsule Networks did not take over the ML world

Capsule Networks are a useful case study because the architecture did **not** disappear for lack of an interesting idea. The original proposal was unusually ambitious: represent entities as vectors, preserve pose-like information rather than destroying it for invariance, learn part-whole transformations, and let lower-level entities dynamically choose higher-level destinations by agreement.

That remains conceptually attractive.

The architecture nevertheless lost the race to become a general foundation for modern machine learning. The best explanation is not one fatal flaw. It is the interaction of **computational cost, optimization difficulty, benchmark/ecosystem reality, weaker-than-hoped semantic guarantees, and the arrival of attention at almost exactly the same historical moment**.

This document separates those factors and also records why some Capsule Network ideas still feel contemporary.

---

## 1. The idea was bigger than "vector-valued neurons"

The 2017 Sabour, Frosst, and Hinton paper defines a capsule as a group of neurons whose activity vector represents the instantiation parameters of an entity such as an object or object part. The vector length is used as an existence probability; its orientation represents learned properties. Lower-level capsules make transformation-based predictions about possible higher-level capsules, and agreement among predictions determines routing.

This is a stronger hypothesis about representation than an ordinary CNN makes.

A CNN feature channel largely asks whether a learned feature occurs at a location. A capsule tries to retain a **stateful entity-like representation** and connect entities through learned transformations.

The attraction is easy to see:

- do not discard pose merely to become invariant to it;
- retain relationships among parts;
- let representations encode richer state than one scalar activation;
- make the network's hierarchy resemble a compositional world of things and their wholes;
- use input-dependent routing so information flow depends on what is present in the current example.

This resembles several research instincts that remain alive in object-centric learning, equivariant models, slot-based representations, sparse routing, world models, and iterative inference.

So "CapsNets failed because the idea was silly" is not a useful history.

---

## 2. What dynamic routing actually computes

For a lower-level capsule \(\mathbf{u}_i\), every candidate parent \(j\) receives a transformed vote:

\[
\hat{\mathbf{u}}_{j|i}=\mathbf{W}_{ij}\mathbf{u}_i.
\]

Routing logits \(b_{ij}\) become coupling coefficients:

\[
c_{ij}=\operatorname{softmax}_j(b_{ij}).
\]

The candidate parent receives a weighted sum:

\[
\mathbf{s}_j=\sum_i c_{ij}\hat{\mathbf{u}}_{j|i},
\]

then is squashed:

\[
\mathbf{v}_j=\operatorname{squash}(\mathbf{s}_j).
\]

Agreement changes routing logits:

\[
b_{ij}\leftarrow b_{ij}+\hat{\mathbf{u}}_{j|i}\cdot\mathbf{v}_j.
\]

Then the softmax/aggregation/agreement loop runs again.

The elegance is that a child does not have one permanently chosen parent. Its current vote is compared with emergent parents, and its routing distribution changes according to agreement on this input.

The cost is that this is an **iterative, densely connected assignment problem embedded inside the forward pass**.

---

# Part I: The scaling problem

## 3. Dense child-parent relationships become expensive quickly

Let:

- \(N\) be the number of lower-level capsules;
- \(M\) be the number of candidate upper-level capsules;
- \(d_{in}\) and \(d_{out}\) be their vector dimensions.

A vanilla relationship can require a matrix

\[
\mathbf{W}_{ij}\in\mathbb{R}^{d_{out}\times d_{in}}
\]

for each \((i,j)\) pair.

That makes the transformation work approximately:

\[
O(NM d_{in}d_{out}).
\]

The routing state/votes also scale with the child-parent relationship graph. Then dynamic routing revisits these relationships for multiple iterations.

This is manageable on MNIST because the world is tiny: small images, ten classes, controlled structure. Scale the number of spatial capsules, object categories, vector dimensions, image resolution, or number of routing layers and the dense relationship graph becomes a very large tax.

The 2025 windowed-routing literature still explicitly identifies quadratic capsule interaction cost as a core problem and proposes constrained/windowed routing to reduce it toward linear order. The persistence of that research problem is evidence that the bottleneck was architectural, not merely a bad 2017 implementation.

Reference: Chen et al., 2025: https://link.springer.com/article/10.1007/s40747-024-01640-8

## 4. Routing is not just expensive arithmetic; it creates synchronization

The original routing algorithm alternates logically dependent stages:

```text
transform votes
-> normalize routing logits
-> reduce weighted votes into parents
-> squash parent vectors
-> compare votes with parents
-> update routing logits
-> repeat
```

Those stages have dependency boundaries. Even if each individual operation parallelizes well, the next routing iteration cannot simply ignore the result of the previous one.

This matters on accelerators. Large dense matrix operations are excellent GPU workloads because a huge amount of arithmetic can happen between synchronization points. Iterative routing contains multiple smaller structured operations and reductions whose intermediate state must be available before the next stage.

This repository's thesis exists largely because mapping those operations efficiently to CUDA was nontrivial. That itself is evidence about the architecture's hardware friction.

## 5. Memory layout was unusually important

An intuitive object-oriented Capsule Network representation wants something like:

```text
parent capsule
  -> list of child votes
  -> one transformation matrix per relationship
  -> routing logits
  -> coupling coefficients
```

That is comfortable for a programmer and awkward for a GPU.

The CUDA thesis implementation had to flatten these structures into contiguous Unified Memory buffers and reconstruct conceptual tensor/vector/matrix coordinates with arithmetic indexing. The speedup was substantial, but the effort required to obtain it is instructive.

Capsule Networks were asking hardware to execute a workload whose **semantic structure was richer than the primitives hardware/software stacks were already optimizing hardest**.

---

# Part II: The optimization problem

## 6. The squash function is conceptually elegant and numerically awkward

The original squash is:

\[
\operatorname{squash}(\mathbf{s}) =
\frac{\|\mathbf{s}\|^2}{1+\|\mathbf{s}\|^2}
\frac{\mathbf{s}}{\|\mathbf{s}\|}.
\]

It creates the attractive interpretation that vector direction carries instantiation information while vector length approaches one for strongly present entities and zero for absent entities.

But saturating nonlinearities create optimization concerns. Later work has explicitly studied gradient-vanishing behavior associated with the squash function and proposed normalization/gradient-friendly alternatives.

The 2025 windowed-routing paper, for example, identifies narrow dynamic range/saturation in squash and sparse gradients from softmax as problems that make deeper Capsule Networks harder to optimize.

Reference: https://link.springer.com/article/10.1007/s40747-024-01640-8

The point is not that squash is mathematically "wrong." The point is that a representation-friendly interpretation does not automatically produce an optimization-friendly deep network.

## 7. Routing itself is another nonlinear optimization system

Softmax coupling coefficients can become sharp, routing iterations can reinforce early assignments, and gradients must pass through an unrolled iterative procedure.

Deeper networks multiply these difficulties. A routing formulation that behaves nicely for one shallow DigitCaps layer does not automatically remain stable when many capsule layers are stacked.

This is one reason later capsule work repeatedly changes the routing algorithm: EM routing, attention routing, self-routing, sparse/windowed routing, and other alternatives are attempts to preserve the representation idea while replacing the troublesome assignment mechanism.

When a research community spends years redesigning the central operator, that operator was probably part of the adoption barrier.

---

# Part III: The benchmark problem

## 8. MNIST was almost the perfect demonstration dataset

The original vector Capsule Network achieved state-of-the-art MNIST performance and showed striking results on overlapping digits. Those are legitimate results. They also occur in a setting especially sympathetic to the architecture's assumptions.

Digits have:

- a small number of semantic classes;
- simple backgrounds;
- coherent object-like structure;
- meaningful pose/deformation variation;
- little texture complexity;
- clean part-whole relationships relative to natural scenes.

The architecture's story is beautifully visible there: parts vote for a coherent digit, and overlapping digits create exactly the kind of assignment problem routing is meant to solve.

Reference: Sabour et al., 2017: https://arxiv.org/abs/1710.09829

## 9. Natural images are messier than the capsule ontology

A realistic image contains texture, background clutter, repeated structures, ambiguous boundaries, partial occlusions, multiple scales, and large numbers of weak or irrelevant local features.

Now lower-level capsules can create many poor votes. The routing mechanism has to distinguish meaningful part-whole agreement from a much noisier candidate set.

The model also needs many more representational units and potentially many more candidate relationships. The same mechanism that is interpretable on digits becomes more expensive and more difficult to optimize on a natural image.

The community never received the decisive large-scale moment that would have justified the complexity. There was no Capsule Network equivalent of:

> "This just crushed ImageNet badly enough that everyone must deal with the routing cost."

Without that result, a complex operator has to compete on engineering convenience as well as conceptual elegance.

## 10. Matrix Capsules improved the story but not the ecosystem outcome

Hinton, Sabour, and Frosst followed with Matrix Capsules and EM Routing. These capsules separated presence probability from a 4x4 pose matrix and used expectation-maximization-like routing to cluster compatible votes. The model reported strong smallNORB results and viewpoint/generalization benefits.

Reference: https://openreview.net/forum?id=HJWLfGWRb

This reinforced that the basic capsule idea had substance. It also increased algorithmic complexity. A research architecture that is already fighting adoption cost rarely benefits from becoming harder to implement, debug, and optimize unless the empirical gains become overwhelming.

---

# Part IV: The representation promise was softer than the story

## 11. A capsule vector is not automatically a human-legible coordinate system

One of the most attractive descriptions of capsules is that vector orientation stores instantiation parameters such as pose.

But ordinary end-to-end training does not force the basis to line up with human concepts.

A coordinate can encode an arbitrary mixture:

\[
z_1 = 0.63(\text{rotation}) + 0.17(\text{x-position}) - 0.42(\text{stroke style}) + \cdots
\]

and still be perfectly useful.

This is not a defect unique to capsules. Learned representations are generally distributed and basis-dependent. It does, however, weaken one rhetorical advantage of Capsules: the promise that the internal vectors are intrinsically more semantically transparent than ordinary hidden representations.

A cautious statement is:

> Capsule vectors can retain structured transformation-sensitive information, but the architecture does not guarantee a clean human-interpretable disentanglement of that information.

Once that guarantee softens, the field is free to ask whether simpler vector representations can learn the same useful information without capsule-specific routing.

## 12. Strong inductive bias is both power and constraint

Capsule Networks assume a world that can profitably be modeled as entity-like units connected through part-whole predictions.

When that assumption matches the task, it can be powerful.

But a general foundation architecture benefits from making fewer assumptions about what every layer must mean. A Transformer head does not require its units to correspond to object parts or pose. It can learn syntax, copying, retrieval, positional relationships, induction-like patterns, visual relationships, or something no human has named.

Capsules offered a stronger ontology.

Transformers offered a more permissive substrate.

At large scale, permissive substrates often win because data and optimization can discover structures that designers did not anticipate.

---

# Part V: CNNs refused to become obsolete

## 13. The critique of CNNs was real, but engineering found other answers

Capsules were motivated partly by weaknesses in pooling/invariance and by the desire to preserve spatial relationships.

The field did not need to prove that critique false in order to move on. CNNs simply became better through other means:

- residual connections;
- normalization;
- stronger optimizers;
- enormous datasets;
- data augmentation;
- multi-scale architectures;
- better detection/segmentation heads;
- self-supervised objectives;
- architectural refinements that preserve more spatial information.

A network can learn robust invariance from data augmentation and architecture without explicitly representing every transformation as a capsule pose relationship.

That may be less philosophically satisfying, but it is operationally powerful.

Capsules therefore had to beat a moving baseline. "CNNs throw away too much geometry" was not enough if increasingly sophisticated CNN systems solved downstream tasks anyway.

---

# Part VI: Attention arrived at exactly the wrong time for CapsNets

## 14. The historical collision

**Attention Is All You Need** appeared in June 2017. **Dynamic Routing Between Capsules** appeared in October 2017.

Both architectures were responding, in very different domains, to a question about **how representations should dynamically exchange information**.

The Transformer answer was extraordinarily accelerator-friendly:

\[
\operatorname{Attention}(Q,K,V)
= \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V.
\]

Conceptually:

```text
GEMM
softmax
GEMM
```

This is not literally the complete implementation cost, but it captures the hardware character: large batched dense linear algebra, exactly what GPU vendors and numerical libraries are heavily motivated to optimize.

The original CapsNet answer looks more like:

```text
transform every vote
normalize assignments
reduce votes into candidate parents
squash vectors
compute agreement
update assignments
synchronize/refine
repeat
```

Both can be parallelized. One maps far more directly onto the accelerator ecosystem.

Vaswani et al. explicitly highlighted that the Transformer was more parallelizable and significantly faster to train than the recurrent/convolutional sequence baselines of the time.

Reference: https://arxiv.org/abs/1706.03762

## 15. The attention analogy is meaningful but not exact

There is a useful family resemblance:

| Dynamic routing | Attention |
|---|---|
| lower capsule/source | token/source representation |
| transformed vote | key/value-like representation conditioned on destination |
| candidate parent | query/prototype-like destination |
| vote-parent dot product | query-key compatibility |
| coupling coefficient | attention weight |
| weighted vote reduction | weighted value aggregation |

Both let vector compatibility determine input-dependent information flow.

But routing is iterative. Candidate parents are partly **created by the votes being routed**, then those emergent parents are used to update assignments. The transformed vote does not split cleanly into the separate Q/K/V roles of ordinary attention.

A good mental model is:

> Dynamic routing is attention-adjacent, but it behaves more like iterative soft clustering/assignment than one standard attention operation.

The 2022 survey **Learning with Capsules** explicitly discusses non-trivial conceptual similarities between capsule routing and Transformer attention.

Reference: https://arxiv.org/abs/2206.02664

## 16. Attention won on weaker assumptions plus stronger scaling

Transformers do not require a hidden vector to be an object part. They do not require a fixed part-whole interpretation for each layer. They do not require repeated routing iterations between every adjacent layer.

They provide a general mechanism:

> compute relevance among representations and move information accordingly.

That weaker structural assumption turned out to be extraordinarily reusable across:

- language;
- vision;
- audio;
- video;
- multimodal models;
- proteins and biological sequences;
- reinforcement-learning/state modeling;
- retrieval and memory systems.

Capsules had a richer story about what a representation **should mean**.

Attention had a simpler story about what a representation **should be allowed to do**.

The latter scaled more cleanly.

### One-sentence version

**Capsule Networks chose a more opinionated representation and a more expensive routing procedure at almost exactly the historical moment when attention showed that a weaker structural assumption, expressed largely as enormous matrix multiplications, could scale absurdly well.**

That is probably the single most important reason they did not take over.

---

# Part VII: The field partially reinvented the useful pieces

## 17. Later Capsule Networks moved toward attention

Efficient-CapsNet is revealing historically. It retains capsule representations but replaces the original iterative dynamic routing with a non-iterative, highly parallelizable self-attention routing mechanism. The authors explicitly frame attention as a way of routing information and report a far smaller operation/parameter footprint than the original CapsNet.

Reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC8290018/

That does not prove that "CapsNet was secretly a Transformer." It shows that later work found the capsule representation idea separable from the original routing algorithm.

That separation may be the most useful way to think about the architecture today:

1. **capsule representation hypothesis**: vector-valued entity/state representations;
2. **part-whole transformation hypothesis**: lower entities predict higher entities;
3. **routing algorithm**: how assignments are inferred;
4. **loss/activation design**: how presence/state are trained.

The original 2017 model bundled all four. Modern experiments can replace one without rejecting the others.

## 18. Object-centric learning still asks capsule-like questions

Current object-centric representation research asks how a scene can be decomposed into interacting entities or "slots." Slot-based methods are not Capsule Networks, but the motivating desire is familiar: represent a scene as structured entities rather than an undifferentiated feature tensor.

Capsules can therefore be viewed as one early, strongly engineered answer to a broader unresolved question:

> How should a neural system bind lower-level evidence into persistent higher-level entities?

That question did not vanish.

## 19. Sparse routing and mixture-of-experts echo another capsule instinct

Mixture-of-experts systems dynamically send tokens/representations to a subset of expert modules. Again, this is not routing-by-agreement, but the broad design principle is familiar:

> the computational path itself can depend on the current input.

Capsules were doing input-dependent routing for representational hierarchy rather than compute sparsity.

## 20. Iterative inference is fashionable again

Dynamic routing spends extra forward-pass computation refining assignments. Modern systems increasingly explore extra inference-time compute, recurrent/iterative refinement, search, verification, and latent reasoning loops.

It would be anachronistic to claim CapsNet routing anticipated all of these systems directly. The useful observation is narrower:

> the idea that a fixed set of learned weights can perform an input-specific inference procedure during the forward pass no longer feels exotic.

Capsules contain a tiny version of that idea.

---

# Part VIII: Why CUDA acceleration helped but could not solve the adoption problem

## 21. This thesis solved a real bottleneck

This repository demonstrates that dynamic routing has significant exploitable parallelism. Custom CUDA kernels and a flattened memory model produced large speedups over the sequential reference implementation.

That is valuable systems work.

But accelerator engineering can improve the constant factors without changing the algorithm's basic scaling or optimization properties.

Even a 30x or 100x improvement over a naive/sequential implementation does not guarantee that the resulting operator beats an alternative architecture whose fundamental primitive is already the best-supported workload in the ecosystem.

The historical comparison also needs context: the thesis baseline was the repository's sequential C++/Armadillo implementation, not a modern optimized GPU framework baseline, and compiler optimization settings were not perfectly symmetric. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## 22. Hardware/software co-evolution matters

Architectures do not compete in a mathematical vacuum.

An algorithm that maps naturally onto:

- GEMM;
- fused elementwise operations;
- tensor cores;
- large batches;
- regular memory access;

inherits enormous investment from accelerator vendors, compiler teams, numerical libraries, and framework developers.

Attention became one of those blessed workloads.

Vanilla dynamic routing asks for a more specialized sequence of transforms, reductions, nonlinear vector normalization, agreement updates, and repeated synchronization.

The thesis can therefore be read as a case study in a broader rule:

> A learning architecture is partly a hardware/software co-design choice, even when the paper describes it only as mathematics.

---

# Part IX: What "failed" should and should not mean

## 23. Capsule Networks did not become the default architecture

That is the practical failure:

- they did not replace CNNs for mainstream computer vision;
- they did not become a general foundation architecture analogous to Transformers;
- standard frameworks/hardware did not converge around capsule-specific primitives;
- research activity became a niche rather than the center of deep learning.

## 24. The representation hypothesis was never decisively falsified

It would be too strong to conclude:

> "Vector-valued entity representations and part-whole routing are bad ideas."

The historical evidence supports a more nuanced conclusion:

- the **original package** of dense transformations + iterative routing + squash + margin loss was hard to scale and optimize;
- the empirical wins did not become large enough on major benchmarks to pay for that complexity;
- simpler/more permissive architectures improved faster;
- attention offered a more general and hardware-compatible routing abstraction.

A component can be interesting while its original system design loses.

## 25. The most durable lesson

CapsNet's most durable question may be more important than its answer:

> **Should a neural model represent the world as an unstructured field of activations, or should it form explicit latent entities and dynamically infer their relationships?**

Transformers proved that surprisingly general intelligence can emerge from relatively weak assumptions plus massive scale. That success does not make structured latent entities irrelevant. It raises the bar for proving when explicit structure is worth its cost.

---

# Part X: Questions worth revisiting in a modern fork

This repository should first be preserved and made reproducible. After that, a modern research branch could ask much sharper questions than were practical in 2018.

## 26. Keep capsule representations, replace routing

Experiments could compare the same capsule state representation under:

- original dynamic routing;
- scaled-dot-product attention routing;
- sparse/top-k routing;
- slot-style iterative assignment;
- EM-style routing;
- learned one-pass routing;
- windowed/local routing.

The key is to isolate **representation choice** from **routing choice**.

## 27. Compare on equal accelerator primitives

A modern systems study should compare:

- hand-written CUDA;
- cuBLAS/cuDNN-based formulations;
- PyTorch compiled/eager variants;
- Triton kernels where useful;
- attention-based routing using optimized attention kernels;
- the same architecture under FP32/TF32/FP16/BF16 as numerically appropriate.

The interesting question is no longer "can CUDA make this faster?" It is:

> "What capsule formulation maps onto modern GPU primitives without surrendering the structured representation hypothesis?"

## 28. Make the latent-state claim measurable

Instead of merely visualizing capsule dimensions, evaluate whether capsule vectors provide measurable advantages for:

- viewpoint extrapolation;
- compositional generalization;
- occlusion;
- object identity under transformations;
- sample efficiency;
- out-of-distribution pose changes;
- calibrated part-whole binding.

If capsules are worth extra routing cost, those are the kinds of wins that should pay the bill.

## 29. Treat routing as inference-time compute

Original routing uses a fixed number of iterations. A modern experiment could ask whether routing depth should be adaptive:

- stop when assignments converge;
- spend more iterations on ambiguous examples;
- use a confidence/entropy criterion;
- compare extra routing compute against a deeper feed-forward baseline at equal FLOPs/latency.

That frames routing in contemporary compute-budget terms.

## 30. Explore sparse part-whole graphs

The fully connected child-parent graph is one source of poor scaling. Modern approaches could learn or constrain candidate parent relationships before expensive transformations.

Possible directions:

- spatial locality/windowing;
- top-k parent proposals;
- coarse-to-fine routing;
- hierarchical routing trees;
- shared transformation families;
- low-rank transformation matrices.

The goal would be to preserve entity composition without paying \(N\times M\) everywhere.

---

# Conclusion

Capsule Networks probably did not take over machine learning because their most distinctive strengths came bundled with costs the rest of the ecosystem had little incentive to absorb.

They offered:

- richer vector-valued state;
- explicit part-whole transformation hypotheses;
- input-dependent iterative assignment;
- an appealing equivariance story.

They also required:

- dense child-parent transformation relationships;
- iterative routing and synchronization;
- specialized vector nonlinearities;
- difficult deep optimization;
- more implementation complexity;
- stronger assumptions about what representations should mean.

Meanwhile, CNNs kept improving, and Transformers arrived with a routing-like abstraction that was less semantically constrained, easier to compose, easier to train deeply, and exceptionally compatible with large dense accelerator operations.

So the useful historical verdict is not:

> "Capsules were wrong."

It is:

> **The original Capsule Network was an elegant but expensive bundle of representation and routing assumptions. Attention captured part of the useful information-routing idea in a much more scalable substrate, while the broader questions about entities, binding, equivariance, and iterative inference remained open.**

That is why the architecture is still worth understanding, even though it did not win.

---

## References and further reading

1. Sabour, S.; Frosst, N.; Hinton, G. **Dynamic Routing Between Capsules** (2017). https://arxiv.org/abs/1710.09829
2. Vaswani, A. et al. **Attention Is All You Need** (2017). https://arxiv.org/abs/1706.03762
3. Hinton, G.; Sabour, S.; Frosst, N. **Matrix Capsules with EM Routing** (ICLR 2018). https://openreview.net/forum?id=HJWLfGWRb
4. Mazzia, V.; Salvetti, F.; Chiaberge, M. **Efficient-CapsNet: capsule network with self-attention routing** (2021). https://pmc.ncbi.nlm.nih.gov/articles/PMC8290018/
5. Ribeiro, F. D. S. et al. **Learning with Capsules: A Survey** (2022). https://arxiv.org/abs/2206.02664
6. Chen, G. et al. **Enhancing classification efficiency in capsule networks through windowed routing: tackling gradient vanishing, dynamic routing, and computational complexity challenges** (published 2024; volume 2025). https://link.springer.com/article/10.1007/s40747-024-01640-8
7. Lopez, D. **A GPU Acceleration Method for Dynamically Routed Capsule Networks**. https://www.cse.unr.edu/~fredh/papers/conf/194-amlfdrcl/paper.pdf
8. Lopez, D. **Evolving GPU-Accelerated Capsule Networks**. https://www.cse.unr.edu/~fredh/papers/thesis/071-lopez/thesis.pdf
