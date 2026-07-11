# TUI Design Guidelines

Use these guidelines for the resistics terminal user interface.

## Colour and Surface States

- Keep the application canvas neutral black (`#101010`) and standard panels
  charcoal (`#202020`). Do not use blue-tinted surfaces for general content.
- Use a lighter neutral grey (`#343434`) to communicate focus, selection, or
  an elevated interactive surface. Prefer this surface-state change to a focus
  border so panels behave consistently.
- Reserve the resistics colours for accents: blue (`#0a009f` and `#070066`) for
  the header and footer, peach (`#faa881`) for active selections and primary
  controls, and orange (`#ac3600`) for successful or destructive actions.
- Keep normal text light (`#f7f4f2`) and disabled text muted (`#aaa6ad`) for
  readable contrast on dark surfaces.

## Interaction

- Every area that can contain more content than fits vertically must be a
  focusable scroll container, not a non-focusable static widget.
- Keyboard users must be able to reach each interactive panel with Tab and
  scroll a focused panel with arrow keys, Page Up/Down, Home, and End.
- Use colour changes to make focus visible; do not rely only on pointer input
  or a border.

## Verification

- Update the focused TUI tests when changing colour or focus behaviour.
- Test Tab order on the active tab, since widgets in inactive tabs are not part
  of normal focus navigation.
