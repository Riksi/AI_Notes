> We now introduce MaskFormer, the new mask classification model, which computes $N$ probability mask pairs $\{(p_i, m_i)\}_{i=1}^N$. The model contains three modules

### Pixel-level module
#### Basic idea
>takes an image of size H × W as input. A backbone generates a (typically)
low-resolution image feature map $\mathcal{F} \in \mathbb{R}
^{C_\mathcal{F} \times \frac{H}{S} \times \frac{W}{S}}$, where $C_\mathcal{F}$ is the number of channels and $S$
is the stride of the feature map ($C_\mathcal{F}$ depends on the specific backbone and we use $S = 32$ in this
work). 


```python
class BasePixelDecoder(nn.Module):
    ...
    def forward_features(self, features):
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if lateral_conv is None:
                y = output_conv(x)
            else:
                cur_fpn = lateral_conv(x)
```



> Then, a pixel decoder gradually upsamples the features to generate per-pixel embeddings
$\mathcal{E}\_\mathrm{pixel} \in \mathbb{R}^
{C_\mathcal{E} \times H \times W}$ , where $C_\mathcal{E}$ is the embedding dimension

```python
            else:
                cur_fpn = lateral_conv(x)
                # Following FPN implementation, we use nearest upsampling here
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)
        return self.mask_features(y), transformer_encoder_features
```

Notes:
- This an [Feature Pyramid Network](FPN.md), except that only the final, highest resolution feature map is used for the mask prediction (the `mask_features` output).
- `self.in_features` comprises the feature maps from the backbone
- There is no lateral convolution in the first layer of the top-down pyramid (`self.lateral_convs[0]` is `None`), so the `else` statement is only executed from the second layer onwards by which time `y` has been initialized to `output_conv(x)`.

#### Transformer-based
> Note, that any per-pixel classificationbased segmentation model fits the pixel-level module design including recent Transformer-based models ... MaskFormer seamlessly converts such a model to mask classification