# Clustering Methods Comparison:

In this project,  I implemented two clustering algorithms, **K-means clustering** and **Spectral clustering** from scratch. I tested these classification methods on two datasets of Cho and Iyer. 

[image](data:image/png;base64,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%0A)

