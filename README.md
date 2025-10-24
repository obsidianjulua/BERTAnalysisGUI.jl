# BERTAnalysisGUI.jl

A GUI interface for BERT-based text analysis using Julia, Blink, and Transformers.

## Features

- Semantic search across documents
- Text similarity comparison
- Sentiment analysis and topic detection
- Text clustering and outlier detection
- Text complexity, coherence, and readability analysis

## Installation

```julia
using Pkg
Pkg.add("BERTAnalysisGUI")
```

Or install from source:

```julia
using Pkg
Pkg.add(url="https://github.com/obsidianjulua/BERTAnalysisGUI.jl")
```

## Usage

```julia
using BERTAnalysisGUI

# Launch the GUI with default BERT model
w = BERTAnalysisGUI.launch_gui()

# Or specify a different model
launch_gui("bert-base-uncased")
```

The GUI will open in a new window where you can interact with various text analysis tools.

## Requirements

- Julia 1.6+
- Blink.jl
- Transformers.jl
- JSON3.jl
- StatsBase.jl

## License

MIT License - see LICENSE file for details
