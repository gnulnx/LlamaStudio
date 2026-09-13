# Live product tour. Run with `lls demo-tui`; the command fills @...@ paths.
# Adjust these timings/navigation after UI changes. No load/eject/download actions.
Output @VIDEO@
Set Shell bash
Set Width 1920
Set Height 1080
Set FontSize 14
Set FontFamily "DejaVu Sans Mono"
Set LineHeight 1.2
Set Padding 18
Set Margin 18
Set MarginFill "#17171b"
Set WindowBar Colorful
Set BorderRadius 10
Set Framerate 30
Set TypingSpeed 45ms
Set Theme {"name":"Studio Charcoal","background":"#202024","foreground":"#e4e4ed","selection":"#454052","cursor":"#a78bfa","black":"#202024","red":"#f87171","green":"#22c55e","yellow":"#eab308","blue":"#8b5cf6","magenta":"#9d6bff","cyan":"#2dd4bf","white":"#e4e4ed","brightBlack":"#717199","brightRed":"#fca5a5","brightGreen":"#4ade80","brightYellow":"#fde047","brightBlue":"#b794f6","brightMagenta":"#c4b5fd","brightCyan":"#5eead4","brightWhite":"#ffffff"}

Hide
Type@1ms @SETUP@
Enter
Sleep 1s
Ctrl+L
Show
Sleep 1s
Type@120ms @COMMAND@
Sleep 500ms
Enter

# Discover: wait for real Hub results, then browse without starting a download.
Wait+Screen@45s /results . Hugging Face GGUF/
Sleep 4s
Screenshot @POSTER@
Down
Sleep 2s
Down
Sleep 2s
Up
Sleep 2s

# Models: browse the local library and saved configuration. No lifecycle changes.
# VHS 0.11 has no function-key commands; activate the ordinary sidebar buttons.
# From the result table: search button, sort, query, Logs, Chat, Models.
Shift+Tab
Shift+Tab
Shift+Tab
Shift+Tab
Shift+Tab
Shift+Tab
Enter
Wait+Screen@30s /LOAD SETTINGS/
Sleep 4s
Down
Sleep 3s

# Chat: the recorder has created a dedicated empty conversation.
# From the local table: rescan, filter, Logs, Chat.
Shift+Tab
Shift+Tab
Shift+Tab
Shift+Tab
Enter
Wait+Screen@30s /Saved conversation . ready/
Sleep 2s
Tab
Space
Shift+Tab
Type "In three short bullets, explain local AI: privacy, offline access, and customization. No tools."
Sleep 700ms
Ctrl+S
Wait+Screen@90s /tokens.s|Response complete/
Sleep 5s

# Logs: real inference activity from the same running model.
# From the composer: transcript, Delete, Rename, conversations, New chat, Logs.
Shift+Tab
Shift+Tab
Shift+Tab
Shift+Tab
Shift+Tab
Shift+Tab
Enter
Wait+Screen@30s /Following/
Sleep 3s
PageUp
Sleep 2s
PageDown
Sleep 2s

# From the log output: Follow, filter, source, Logs, Chat, Models, Discover.
Shift+Tab
Shift+Tab
Shift+Tab
Shift+Tab
Shift+Tab
Shift+Tab
Shift+Tab
Enter
Up
Sleep 4s
Ctrl+Q
Sleep 2s
