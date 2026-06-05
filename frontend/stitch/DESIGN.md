---
name: Synthesized Intelligence
colors:
  surface: '#0b1326'
  surface-dim: '#0b1326'
  surface-bright: '#31394d'
  surface-container-lowest: '#060e20'
  surface-container-low: '#131b2e'
  surface-container: '#171f33'
  surface-container-high: '#222a3d'
  surface-container-highest: '#2d3449'
  on-surface: '#dae2fd'
  on-surface-variant: '#c2c6d6'
  inverse-surface: '#dae2fd'
  inverse-on-surface: '#283044'
  outline: '#8c909f'
  outline-variant: '#424754'
  surface-tint: '#adc6ff'
  primary: '#adc6ff'
  on-primary: '#002e6a'
  primary-container: '#4d8eff'
  on-primary-container: '#00285d'
  inverse-primary: '#005ac2'
  secondary: '#b7c8e1'
  on-secondary: '#213145'
  secondary-container: '#3a4a5f'
  on-secondary-container: '#a9bad3'
  tertiary: '#89ceff'
  on-tertiary: '#00344d'
  tertiary-container: '#009ada'
  on-tertiary-container: '#002d43'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#d8e2ff'
  primary-fixed-dim: '#adc6ff'
  on-primary-fixed: '#001a42'
  on-primary-fixed-variant: '#004395'
  secondary-fixed: '#d3e4fe'
  secondary-fixed-dim: '#b7c8e1'
  on-secondary-fixed: '#0b1c30'
  on-secondary-fixed-variant: '#38485d'
  tertiary-fixed: '#c9e6ff'
  tertiary-fixed-dim: '#89ceff'
  on-tertiary-fixed: '#001e2f'
  on-tertiary-fixed-variant: '#004c6e'
  background: '#0b1326'
  on-background: '#dae2fd'
  surface-variant: '#2d3449'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.01em
  headline-lg-mobile:
    fontFamily: Inter
    fontSize: 28px
    fontWeight: '600'
    lineHeight: 36px
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-sm:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  label-md:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.05em
  code-snippet:
    fontFamily: JetBrains Mono
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 22px
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  base: 4px
  xs: 0.25rem
  sm: 0.5rem
  md: 1rem
  lg: 1.5rem
  xl: 2rem
  gutter: 1.5rem
  margin-mobile: 1rem
  margin-desktop: 2.5rem
  max-content-width: 1280px
---

## Brand & Style
This design system is engineered for deep focus and academic rigor. The brand personality is intellectual, precise, and authoritative, catering to researchers and analysts who require a low-friction environment for complex data synthesis.

The design style is **Minimalist-Modern with Tonal Layering**. It leverages a deep dark-mode palette to reduce eye strain during extended research sessions. The aesthetic prioritizes structural clarity over decorative elements, using subtle translucency and precise borders to create a sense of organized depth without distracting the user from the content.

## Colors
The palette is rooted in a "Deep Slate" ecosystem. The foundation is `#0f172a` for primary backgrounds, while `#1e293b` serves as the surface color for cards and elevated containers. 

Vibrant Blue (`#3b82f6`) is reserved strictly for primary interactive states and critical progress indicators. Semantic colors (success, warning, error) should be desaturated to maintain the sophisticated atmosphere, only gaining vibrancy on hover or active states. Text colors must maintain a high contrast ratio against the slate background, primarily using off-whites and cool greys to ensure long-form readability.

## Typography
The system utilizes **Inter** for all primary communication to ensure maximum legibility across different display densities. For technical metadata, citations, and AI-generated code snippets, **JetBrains Mono** is introduced to provide a clear visual distinction between narrative content and data.

Tracking is slightly tightened on headlines to create a premium, "locked-in" look. Body text uses a generous line-height (1.5x) to prevent fatigue during heavy reading. Large display sizes are scaled down on mobile to ensure headings remain within the viewport without awkward wrapping.

## Layout & Spacing
The layout follows a **Fluid Grid** model with a maximum content width to preserve line-length readability for research papers. 

- **Desktop:** 12-column grid with 24px (1.5rem) gutters. Margins are expansive at 40px (2.5rem) to provide breathing room.
- **Tablet:** 8-column grid with 20px gutters.
- **Mobile:** 4-column grid with 16px (1rem) gutters and margins.

Spacing follows a strict 4px base unit. Component internal padding should be generous; for example, cards should default to `lg` (24px) padding to reflect the "sophisticated" requirement. Section vertical spacing should utilize `xl` (32px) or higher to clearly delineate different research blocks.

## Elevation & Depth
Depth is communicated through **Tonal Layers** rather than heavy drop shadows. 

1. **Level 0 (Base):** `#0f172a` — The main canvas.
2. **Level 1 (Cards/Sidebar):** `#1e293b` — Primary containers.
3. **Level 2 (Modals/Popovers):** `#334155` — Floating elements.

Borders are the primary tool for definition. Use a 1px solid border (`rgba(255, 255, 255, 0.1)`) on all containers. For interactive elements like cards, a subtle ambient shadow (Blur 12px, Y 4px, 20% opacity black) should only appear on hover to indicate tactility.

## Shapes
The shape language is **Soft** and professional. A standard radius of `4px` (0.25rem) is applied to most UI components including buttons, input fields, and small chips. Larger containers like cards or content modules use `rounded-lg` (8px) to soften the interface without appearing overly "bubbly" or casual. This balance maintains the research-focused, serious tone of the tool.

## Components
- **Buttons:** Follow Bootstrap 5 structures but with a 1px border. The primary button is a solid fill of `#3b82f6` with white text. Secondary buttons use a ghost style (transparent fill) with a `#64748b` border.
- **Input Fields:** Background should be a shade darker than the surface it sits on. Use a 1px border that glows with a subtle blue outer shadow only when focused.
- **Chips/Badges:** Small, `label-md` typography. Backgrounds should be low-contrast (e.g., `#334155`) with the text color being a lighter version of the category color.
- **Cards:** No box-shadow in default state; 1px border of `rgba(255, 255, 255, 0.1)`. Headers within cards should be separated by a subtle horizontal rule.
- **Citations/Footnotes:** Use a distinct background (Level 2 depth) and JetBrains Mono at `body-sm` size.
- **Scrollbars:** Custom-styled to be thin and dark (`#334155`) to avoid breaking the dark-mode immersion.