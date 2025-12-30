# Frontend Changes: Dark/Light Theme Toggle

## Summary
Added a theme toggle button that allows users to switch between dark and light themes with smooth transitions and localStorage persistence.

## Files Modified

### 1. `frontend/index.html`
- Added a theme toggle button with sun/moon SVG icons positioned in the top-right corner
- Button includes proper accessibility attributes (`aria-label`, `title`)
- Uses two icons: sun icon (visible in dark mode) and moon icon (visible in light mode)

### 2. `frontend/style.css`

#### New CSS Variables
Added new CSS variables for better theme control:
- `--code-bg`: Background color for code blocks
- `--scrollbar-thumb`: Scrollbar thumb color
- `--scrollbar-thumb-hover`: Scrollbar thumb hover color

#### Light Theme Definition
Added `[data-theme="light"]` selector with appropriate light theme colors:
- Light background (`#f8fafc`) and white surfaces (`#ffffff`)
- Dark text for good contrast (`#1e293b` primary, `#64748b` secondary)
- Lighter borders (`#e2e8f0`)
- Adjusted shadows for light backgrounds
- Light welcome message background (`#eff6ff`)

#### Theme Toggle Button Styles
- Fixed position in top-right corner (`top: 1rem`, `right: 1rem`)
- 44px circular button with z-index 1000
- Hover effects with scale transform and border color change
- Focus ring for keyboard accessibility
- Icon visibility toggling based on current theme

#### Smooth Transitions
Added `transition: 0.3s ease` to the following elements for smooth theme switching:
- `body` (background-color, color)
- `.sidebar` (background-color, border-color)
- `.message-content` (background-color, color)
- `.chat-input-container` (background-color, border-color)
- `#chatInput` (all properties)
- `.suggested-item` (all properties)
- `.stat-item` (background-color, border-color)
- `.clear-chat-button` (all properties)

#### Updated Variable Usage
- Updated `.message-content code` and `.message-content pre` to use `--code-bg`
- Updated scrollbar thumbs to use `--scrollbar-thumb` and `--scrollbar-thumb-hover`

### 3. `frontend/script.js`

#### New DOM Element
- Added `themeToggle` variable to track the theme toggle button

#### New Functions
- `initializeTheme()`: Checks localStorage for saved theme preference, defaults to dark mode
- `toggleTheme()`: Toggles between dark and light themes, updates `data-theme` attribute on `<html>` and saves preference to localStorage

#### Event Listener
- Added click event listener on theme toggle button to call `toggleTheme()`

## Features

1. **Toggle Button Design**
   - Circular button (44px) positioned in top-right corner
   - Sun icon displayed in dark mode, moon icon in light mode
   - Smooth hover animations (scale, border color change)
   - Keyboard accessible with visible focus ring

2. **Theme Persistence**
   - Theme preference saved to localStorage
   - Persists across page reloads and sessions
   - Defaults to dark theme for new users

3. **Smooth Transitions**
   - 0.3s ease transitions on all themed elements
   - No jarring color changes when toggling

4. **Accessibility**
   - `aria-label="Toggle dark/light theme"` for screen readers
   - `title="Toggle theme"` for tooltip
   - Focus visible styles for keyboard navigation
   - Good color contrast maintained in both themes

## Color Palette

### Dark Theme (Default)
| Variable | Value | Description |
|----------|-------|-------------|
| `--background` | `#0f172a` | Main background |
| `--surface` | `#1e293b` | Card/surface background |
| `--text-primary` | `#f1f5f9` | Primary text |
| `--text-secondary` | `#94a3b8` | Secondary text |
| `--border-color` | `#334155` | Borders |

### Light Theme
| Variable | Value | Description |
|----------|-------|-------------|
| `--background` | `#f8fafc` | Main background |
| `--surface` | `#ffffff` | Card/surface background |
| `--text-primary` | `#1e293b` | Primary text |
| `--text-secondary` | `#64748b` | Secondary text |
| `--border-color` | `#e2e8f0` | Borders |
