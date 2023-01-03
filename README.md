# MMultiBART for long contex

## Arhitecture and motivation

Multiple Encoder/ Multiple Decoder type transformer model for BART, designed for long context tasks(especially finedtune for summarization). The multiple decoder option aims to improve the efficiency adn accuracy when dealing with long target sequences.

The input/target is chunked/preproccesed/labeld and send through the model.
<pre>
[Tokenized Chunk] -> Encoder |
[Tokenized Chunk] -> Encoder |                            [Embedded chunk] -> Decoder |
[Tokenized Chunk] -> Encoder | -> [Concat embeddings] ->  [Embedded chunk] -> Decoder | -> [Generated sequence]
...                          |                            [Embedded chunk] -> Decoder |
[Tokenized Chunk] -> Encoder | 
</pre>