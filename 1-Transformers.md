# Chapter 1: Transformers

The transformer is an important neural network architecture used for language modeling.

## Recommended reading

- [x] [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Section 3 of the paper that introduced the transformer explains the architecture. Don't worry too much about the encoder and how that fits in, as that's somewhat specific to translation – unsupervised transformer language models are generally decoder-only.

## Optional reading

- [x] [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - A blog post explaining the architecture more carefully. Read this if you're finding the original paper hard to follow.
- [ ] [GPT-3](https://arxiv.org/abs/2005.14165) - A 175-billion parameter decoder-only transformer language model that exhibits impressive meta-learning capabilities.
- [ ] [The Transformer Family](https://lilianweng.github.io/posts/2020-04-07-the-transformer-family/) - An overview of many transformer variants, including Transformer-XL, Image Transformer, Sparse Transformer, Reformer and Universal Transformer.
- [ ] [T5](https://arxiv.org/abs/1910.10683) - A careful study of different architectural details and training objectives for transformers.
- [ ] [Mixture-of-Experts](https://arxiv.org/abs/1701.06538) - A form of parameter sparsity used by some more recent language models to improve training efficiency. Section 2 of this paper explains how they work.

## Suggested exercise

Implement a decoder-only transformer language model.

- [x] Here are some first principle questions to answer:
  - What is different architecturally from the Transformer, vs a normal RNN, like an LSTM? (Specifically, how are recurrence and time managed?)
    - Transformers process an entire sequence of N tokens in one forward pass. RNNs need to process it sequentially, one token after another. This makes training transformers parallelizable.
  - Attention is defined as, Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V. What are the dimensions for Q, K, and V? Why do we use this setup? What other combinations could we do with (Q,K) that also output weights?
    - The dimensions for each q, k, and v are: `batch x pos x d_head` with `d_head = d_model / n_heads`.
  - Are the dense layers different at each multi-head attention block? Why or why not? 
    - They have the same dimensions because they read from and write to vector spaces of the same dimensionality (`d_head` and `d_model`, respectively).
  - Why do we have so many skip connections, especially connecting the input of an attention function to the output? Intuitively, what if we didn't? 
    - Residual connections ensure that, in principle, no information is lost when passing through a layer. Without residual connections, every layer would be rewriting the entire content of the residual stream. With residual connections, the layer only reads from it and writes something back (based on what's been read and computed based on the weights).
- [x] Now we'll actually implement the code. Make sure each of these is completely correct - it's very easy to get the small details wrong.
  - Implement the positional embedding function first. 
  - Then implement the function which calculates attention, given (Q,K,V) as arguments. 
  - Now implement the masking function. 
  - Put it all together to form an entire attention block. 
  - Finish the whole architecture.
  - If you get stuck, [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) may help, but don't just copy-paste the code.
- [x] To check you have the attention mask set up correctly, train your model on a toy task, such as reversing a random sequence of tokens. The model should be able to predict the second half of the sequence, but not the first.
- [x] Finally, train your model on [the complete works of William Shakespeare](https://www.gutenberg.org/files/100/100-0.txt). Tokenize the corpus by splitting at word boundaries (`re.split(r"\b", ...)`). Make sure you don't use overlapping sequences as this can lead to overfitting.
