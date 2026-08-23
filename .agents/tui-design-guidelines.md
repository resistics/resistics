# TUI Design Guidelines

Use these guidelines for the resistics terminal user interface.

## Colour and Surface States

- Keep the application canvas neutral black (`#101010`) and standard panels
  charcoal (`#202020`). Do not use blue-tinted surfaces for general content.
- Use a lighter neutral grey (`#343434`) to communicate selection or an elevated
  interactive surface. Keep focused content-panel surfaces charcoal rather than
  tinting their backgrounds.
- Frame content panels with a native heavy border: neutral grey (`#555555`) at
  rest and deep blue (`#003054`) when focused. Keep the border thickness fixed
  across focus changes so content and scrollbars remain inside the frame.
- Retain rounded borders for dialog frames, tall borders for single-line inputs,
  and borderless surface styling for buttons. Consistency applies within these
  interface roles rather than requiring one border style everywhere.
- Use button colour only to communicate interaction state: neutral grey
  (`#343434`) when unselected, soft blue (`#003054`) when focused, and charcoal
  (`#202020`) with muted text when disabled. Button purpose must not change its
  colour, including during hover and press interactions.
- Reserve strong blue (`#0a009f` and `#070066`) for the header and footer. Use
  soft blue (`#003054`) for focus frames, selected buttons, and scrollbars.
- Keep peach and orange accents for non-button selections, warnings, and
  destructive dialog frames where semantic context remains useful.
- Keep normal text light (`#f7f4f2`) and disabled text muted (`#aaa6ad`) for
  readable contrast on dark surfaces.

## Interaction

- Every area that can contain more content than fits vertically must be a
  focusable scroll container, not a non-focusable static widget.
- Keyboard users must be able to reach each interactive panel with Tab and
  scroll a focused panel with arrow keys, Page Up/Down, Home, and End.
- Choice menus should support Up/Down navigation in addition to Tab, using a
  visible focus state and cyclic ordering where it helps rapid selection.
- File pickers should start in the user's home directory and provide an
  explicit keyboard-accessible action to move to the parent directory.
- Combine focus colour with a non-colour cue such as bold button labels; do not
  rely only on pointer input or colour.

## Verification

- Update the focused TUI tests when changing colour or focus behaviour.
- Test Tab order on the active tab, since widgets in inactive tabs are not part
  of normal focus navigation.
