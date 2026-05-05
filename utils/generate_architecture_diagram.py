#!/usr/bin/env python3
"""Generate momo-llm model architecture diagram.
Usage: python3 utils/generate_architecture_diagram.py
Output: model_architecture.png (project root)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(22, 15), facecolor='white')
ax.set_xlim(0, 22)
ax.set_ylim(0, 15)
ax.axis('off')

# Color palettes: (fill, edge)
C_IO    = ('#fce4ec', '#c62828')
C_EMBED = ('#f3e5f5', '#7b1fa2')
C_NORM  = ('#e8f5e9', '#2e7d32')
C_ATTN  = ('#e3f2fd', '#1565c0')
C_FFN   = ('#fff8e1', '#f57f17')
C_ACT   = ('#fbe9e7', '#bf360c')
C_MISC  = ('#f5f5f5', '#757575')
C_DROP  = ('#fafafa', '#9e9e9e')
C_GATE  = ('#fff3e0', '#e65100')


def box(cx, cy, w=4.0, text='', subtext=None, h=0.52,
        fc='#e3f2fd', ec='#1565c0', tc='#1a1a2e',
        fs=8.5, bold=False, dashed=False):
    ls = (0, (5, 2)) if dashed else '-'
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle='round,pad=0.06',
                       facecolor=fc, edgecolor=ec,
                       linewidth=1.5, linestyle=ls, zorder=2)
    ax.add_patch(p)
    fw = 'bold' if bold else 'normal'
    ty = cy + (0.12 if subtext else 0)
    ax.text(cx, ty, text, ha='center', va='center', fontsize=fs,
            fontweight=fw, color=tc, zorder=3)
    if subtext:
        ax.text(cx, cy - 0.13, subtext, ha='center', va='center',
                fontsize=fs - 1.5, color='#666', style='italic', zorder=3)


def sublabel(cx, cy, w, text):
    """Small italic label floated above a box, used for sub-layer markers."""
    ax.text(cx - w / 2, cy + 0.33, text, ha='left', va='center',
            fontsize=7, color='#78909c', style='italic', zorder=3)


def darr(x, y1, y2):
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='->', color='#424242', lw=1.3), zorder=1)


def harr(x1, x2, y, color='#888888'):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2), zorder=1)


def seg(xs, ys, color='#888888', lw=1.2):
    ax.plot(xs, ys, color=color, lw=lw, zorder=1)


def circ(cx, cy, r=0.24, text='', fc='#f5f5f5', ec='#757575', fs=10, tc='#333'):
    ax.add_patch(plt.Circle((cx, cy), r,
                             facecolor=fc, edgecolor=ec, linewidth=1.3, zorder=2))
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fs, color=tc, zorder=3)


def skip_rail(x_rail, y_top, y_bot, cx, r=0.24, label='shortcut'):
    """Draw a residual skip connection: branch right, rail down, arrow left into ⊕."""
    seg([cx, x_rail], [y_top, y_top])
    seg([x_rail, x_rail], [y_top, y_bot])
    harr(x_rail, cx + r, y_bot)
    mid = (y_top + y_bot) / 2
    ax.text(x_rail + 0.09, mid, label, fontsize=6.5, color='#9e9e9e',
            va='center', ha='left', rotation=90, style='italic', zorder=3)


def panel_title(cx, cy, text):
    ax.text(cx, cy, text, ha='center', va='center', fontsize=10.5,
            fontweight='bold', color='#212121',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#eceff1',
                      edgecolor='#90a4ae', lw=1.5), zorder=3)


def vdiv(x, y0=0.3, y1=14.82):
    ax.plot([x, x], [y0, y1], color='#cfd8dc', lw=1.5, ls='--', zorder=0)


def hdiv(x0, x1, y):
    ax.plot([x0, x1], [y, y], color='#cfd8dc', lw=1.5, ls='--', zorder=0)


def sublayer_bg(cx, w, y_top, y_bot, label):
    """Faint background band grouping a sub-layer, with a side label."""
    pad = 0.08
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2 - pad, y_bot - pad), w + 2 * pad, (y_top - y_bot) + 2 * pad,
        boxstyle='round,pad=0.0', facecolor='#f0f4f8', edgecolor='#b0bec5',
        linewidth=0.8, linestyle=(0, (3, 3)), zorder=0, alpha=0.6))
    ax.text(cx - w / 2 - pad - 0.06, (y_top + y_bot) / 2, label,
            ha='right', va='center', fontsize=7, color='#607d8b',
            style='italic', rotation=90, zorder=3)


# ─────────────────────────────────────────────────────────────
# PANEL 1  MomoModel
# ─────────────────────────────────────────────────────────────
P1, W1 = 2.8, 4.8
panel_title(P1, 14.68, 'MomoModel')

box(P1, 14.0, W1, 'Input Tokens', fc=C_IO[0], ec=C_IO[1], bold=True)
darr(P1, 14.0 - 0.26, 13.0 + 0.28)

box(P1, 13.0, W1, 'Token Embedding',
    'nn.Embedding(VOCABULARY_SIZE × EMBEDDING_DIMENSION)',
    fc=C_EMBED[0], ec=C_EMBED[1])
darr(P1, 13.0 - 0.28, 12.0 + 0.28)

box(P1, 12.0, W1, 'Embedding Dropout  (optional)',
    'nn.Dropout(DROPOUT_EMBEDDING_RATE)',
    fc=C_DROP[0], ec=C_DROP[1], tc='#757575', dashed=True)
darr(P1, 12.0 - 0.28, 11.28 + 0.46)

# × N_LAYERS dashed container
blk_cy, blk_h = 10.8, 1.0
ax.add_patch(FancyBboxPatch(
    (P1 - W1 / 2 - 0.15, blk_cy - blk_h / 2),
    W1 + 0.3, blk_h,
    boxstyle='round,pad=0.06', facecolor='#fafafa',
    edgecolor='#90a4ae', linewidth=1.5, ls=(0, (5, 2)), zorder=1))
ax.text(P1 + W1 / 2 + 0.12, blk_cy + blk_h / 2,
        '× N_LAYERS', ha='right', va='bottom',
        fontsize=8, color='#546e7a', style='italic')
box(P1, blk_cy + 0.15, W1 - 0.3, 'MomoTransformerBlock', fc=C_ATTN[0], ec=C_ATTN[1])
ax.text(P1, blk_cy - 0.28, '(see panel 2)',
        ha='center', va='center', fontsize=7.5, color='#555', style='italic')

darr(P1, blk_cy - blk_h / 2 - 0.02, 9.55 + 0.28)

box(P1, 9.55, W1, 'Final RMSNormalization', 'RMSNorm(d)',
    fc=C_NORM[0], ec=C_NORM[1])
darr(P1, 9.55 - 0.28, 8.55 + 0.28)

box(P1, 8.55, W1, 'Output Head',
    'nn.Linear(d → VOCABULARY_SIZE,  bias=False)',
    fc=C_ATTN[0], ec=C_ATTN[1])
darr(P1, 8.55 - 0.28, 7.55 + 0.28)

box(P1, 7.55, W1, 'Logits',
    '(batch × sequence × vocab)',
    fc=C_IO[0], ec=C_IO[1], bold=True)

# ─────────────────────────────────────────────────────────────
# PANEL 2  MomoTransformerBlock
# ─────────────────────────────────────────────────────────────
vdiv(5.8)
P2, W2 = 9.1, 4.4
SK = P2 + W2 / 2 + 0.65

panel_title(P2, 14.68, 'MomoTransformerBlock')

in_y  = 14.0
n1_y  = 13.13   # pre-norm  (layer=None, scale init = 1)
mh_y  = 12.23   # attention
n2_y  = 11.36   # post-norm (layer=L,    scale init = 1/√layer)
d1_y  = 10.49   # dropout
a1_y  =  9.62   # add ⊕
n3_y  =  8.73   # pre-norm  (layer=None, scale init = 1)
ff_y  =  7.83   # FFN
n4_y  =  6.96   # post-norm (layer=L,    scale init = 1/√layer)
d2_y  =  6.09   # dropout
a2_y  =  5.22   # add ⊕
out_y =  4.32   # output

box(P2, in_y, W2, 'x  (input)', fc=C_IO[0], ec=C_IO[1], bold=True)

# Sub-layer 1: Attention
sublayer_bg(P2, W2, n1_y + 0.5, d1_y - 0.5, 'Sub-layer 1')
darr(P2, in_y - 0.26, n1_y + 0.26)
box(P2, n1_y, W2, 'RMSNormalization', 'normalizationLayer1  —  pre-norm  (scale init: 1)',
    fc=C_NORM[0], ec=C_NORM[1])
darr(P2, n1_y - 0.26, mh_y + 0.26)
box(P2, mh_y, W2, 'MultiHeadAttention', '(see panel 3)',
    fc=C_ATTN[0], ec=C_ATTN[1])
darr(P2, mh_y - 0.26, n2_y + 0.26)
box(P2, n2_y, W2, 'RMSNormalization', 'normalizationLayer2  —  post-norm  (scale init: 1/√layer)',
    fc=C_NORM[0], ec=C_NORM[1])
darr(P2, n2_y - 0.26, d1_y + 0.26)
box(P2, d1_y, W2, 'Shortcut Dropout  (optional)', 'dropoutShortcut',
    fc=C_DROP[0], ec=C_DROP[1], tc='#757575', dashed=True)
darr(P2, d1_y - 0.26, a1_y + 0.24)
circ(P2, a1_y, text='⊕')
skip_rail(SK, in_y - 0.26, a1_y, P2)

# Sub-layer 2: FFN
sublayer_bg(P2, W2, n3_y + 0.5, d2_y - 0.5, 'Sub-layer 2')
darr(P2, a1_y - 0.24, n3_y + 0.26)
box(P2, n3_y, W2, 'RMSNormalization', 'normalizationLayer3  —  pre-norm  (scale init: 1)',
    fc=C_NORM[0], ec=C_NORM[1])
darr(P2, n3_y - 0.26, ff_y + 0.26)
box(P2, ff_y, W2, 'FeedForwardBypass', '(see panel 4)',
    fc=C_FFN[0], ec=C_FFN[1])
darr(P2, ff_y - 0.26, n4_y + 0.26)
box(P2, n4_y, W2, 'RMSNormalization', 'normalizationLayer4  —  post-norm  (scale init: 1/√layer)',
    fc=C_NORM[0], ec=C_NORM[1])
darr(P2, n4_y - 0.26, d2_y + 0.26)
box(P2, d2_y, W2, 'Shortcut Dropout  (optional)', 'dropoutShortcut',
    fc=C_DROP[0], ec=C_DROP[1], tc='#757575', dashed=True)
darr(P2, d2_y - 0.26, a2_y + 0.24)
circ(P2, a2_y, text='⊕')
skip_rail(SK, a1_y - 0.24, a2_y, P2)

darr(P2, a2_y - 0.24, out_y + 0.26)
box(P2, out_y, W2, 'x  (output)', fc=C_IO[0], ec=C_IO[1], bold=True)

# ─────────────────────────────────────────────────────────────
# PANEL 3  MultiHeadAttention  (top-right, y > 4.8)
# ─────────────────────────────────────────────────────────────
vdiv(12.5)
P3, W3 = 16.9, 5.5
panel_title(P3, 14.68, 'MultiHeadAttention')

# x_orig label: clarify gate uses original x
mha_rows = [
    (14.0,  'x  (batch × seq × d)',
     None,
     C_IO, True, False),
    (13.07, 'wQuery  |  wKey  |  wValue',
     '3 × nn.Linear(d → d,  bias=QKV_BIAS)',
     C_ATTN, False, False),
    (12.13, 'Reshape  →  N_HEADS × head_dim',
     'view(batch, seq, N_HEADS, d/N_HEADS)  →  transpose',
     C_MISC, False, False),
    (11.18, 'Attention Scores  =  Q @ Kᵀ',
     None,
     C_ATTN, False, False),
    (10.23, 'Causal Mask  (−∞ on upper triangle)',
     'triu(ones(ctx, ctx), diagonal=1)  —  registered buffer',
     C_MISC, False, False),
    (9.28,  'Softmax  ( scores / √head_dim )',
     None,
     C_ATTN, False, False),
    (8.33,  'Attention Dropout  (optional)',
     'nn.Dropout(DROPOUT_ATTENTION_RATE)',
     C_DROP, False, True),
    (7.38,  'Context  =  attn_weights @ V',
     'reshape  →  (batch × seq × d)',
     C_ATTN, False, False),
    (6.43,  '[optional]  context × sigmoid( wGate(x) )',
     'GATED_ATTENTION=True  —  gate uses original x, not context',
     C_GATE, False, True),
    (5.48,  'Output Projection',
     'nn.Linear(d → d)',
     C_ATTN, False, False),
]

prev_y = None
for y, text, subtext, col, bold, dashed in mha_rows:
    if bold:
        tc = '#1a1a2e'
    elif col is C_GATE:
        tc = '#e65100'
    elif dashed:
        tc = '#757575'
    else:
        tc = '#1a1a2e'
    box(P3, y, W3, text, subtext, fc=col[0], ec=col[1],
        tc=tc, bold=bold, dashed=dashed)
    if prev_y is not None:
        darr(P3, prev_y - 0.26, y + 0.26)
    prev_y = y

# KV-cache note beside the QKV row
ax.annotate('KV cache\n(useCache=True)',
            xy=(P3 + W3 / 2 + 0.08, 7.38),
            xytext=(P3 + W3 / 2 + 1.2, 8.2),
            fontsize=6.8, color='#1565c0', ha='left', va='center',
            style='italic',
            arrowprops=dict(arrowstyle='->', color='#90a4ae', lw=1.0,
                            connectionstyle='arc3,rad=-0.25'))

# ─────────────────────────────────────────────────────────────
# PANEL 4  FeedForwardBypass  (bottom-right, y < 4.8)
# ─────────────────────────────────────────────────────────────
hdiv(12.5, 22.0, 4.9)
P4, W4 = 16.9, 5.5
panel_title(P4, 4.58, 'FeedForwardBypass  (SwiGLU-style)')

LW  = 2.3
L1x = P4 - 1.45
L2x = P4 + 1.45
fi_y, l_y, sw_y, mu_y, l3_y = 4.08, 3.18, 2.25, 1.48, 0.62

box(P4, fi_y, W4, 'x  (input)', fc=C_IO[0], ec=C_IO[1], bold=True)

# branch: input → layer1 & layer2 in parallel
bx = fi_y - 0.26 - 0.2
seg([P4, P4],    [fi_y - 0.26, bx])
seg([L1x, L2x],  [bx, bx])
darr(L1x, bx, l_y + 0.26)
darr(L2x, bx, l_y + 0.26)

box(L1x, l_y, LW, 'layer1(x)', 'nn.Linear(d→d)', fc=C_FFN[0], ec=C_FFN[1], fs=8.2)
box(L2x, l_y, LW, 'layer2(x)', 'nn.Linear(d→d)', fc=C_FFN[0], ec=C_FFN[1], fs=8.2)

darr(L1x, l_y - 0.26, sw_y + 0.26)
box(L1x, sw_y, LW, 'Swish  activation', 'x · sigmoid(x)',
    fc=C_ACT[0], ec=C_ACT[1], fs=8.2)

circ(P4, mu_y, text='×')

# swish → multiply (down, then right to circle left edge)
seg([L1x, L1x], [sw_y - 0.26, mu_y])
harr(L1x, P4 - 0.24, mu_y, color='#424242')

# layer2 → multiply (down, then left to circle right edge)
seg([L2x, L2x], [l_y - 0.26, mu_y])
harr(L2x, P4 + 0.24, mu_y, color='#424242')

darr(P4, mu_y - 0.24, l3_y + 0.26)
box(P4, l3_y, W4, 'layer3  (output projection)', 'nn.Linear(d→d)',
    fc=C_FFN[0], ec=C_FFN[1])

# ─────────────────────────────────────────────────────────────
# Legend  (bottom-left)
# ─────────────────────────────────────────────────────────────
lx, ly = 0.18, 5.5
ax.text(lx + 0.05, ly + 0.06, 'Legend',
        fontsize=9, fontweight='bold', color='#333')
legend_items = [
    (*C_IO,    'Input / Output'),
    (*C_EMBED, 'Embedding'),
    (*C_NORM,  'Normalization (RMS)'),
    (*C_ATTN,  'Attention / Linear'),
    (*C_FFN,   'Feed-Forward'),
    (*C_ACT,   'Activation (Swish)'),
    (*C_GATE,  'Optional gating'),
    (*C_DROP,  'Optional / Dropout'),
]
for i, (fc, ec, label) in enumerate(legend_items):
    ry = ly - 0.54 - i * 0.52
    ax.add_patch(FancyBboxPatch(
        (lx, ry - 0.14), 0.38, 0.30,
        boxstyle='round,pad=0.03',
        facecolor=fc, edgecolor=ec, lw=1.2, zorder=2))
    ax.text(lx + 0.52, ry + 0.01, label,
            va='center', fontsize=8, color='#333', zorder=3)

plt.savefig('model_architecture.png', dpi=130,
            bbox_inches='tight', facecolor='white', edgecolor='none')
print('Saved model_architecture.png')
