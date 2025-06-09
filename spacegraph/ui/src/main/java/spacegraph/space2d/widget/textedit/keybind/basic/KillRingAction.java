package spacegraph.space2d.widget.textedit.keybind.basic;

import spacegraph.space2d.widget.textedit.TextEditModel;
import spacegraph.space2d.widget.textedit.buffer.Buffer;
import spacegraph.space2d.widget.textedit.keybind.Action;

import java.awt.*;
import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.StringSelection;

public class KillRingAction implements Action {

  @Override
  public String name() {
    return "kill-ring";
  }

  @Override
  public void execute(TextEditModel editor, String... args) {
    Buffer currentBuffer = editor.buffer();
    if (currentBuffer.isLineEnd()) {
      currentBuffer.delete();
    } else {
      currentBuffer.mark();
      currentBuffer.last();
      setClipboardString(currentBuffer.copy());
      currentBuffer.cut();
    }
  }

  private static void setClipboardString(String value) {
    StringSelection selection = new StringSelection(value);

    Toolkit toolKit = Toolkit.getDefaultToolkit();
    Clipboard clipboard = toolKit.getSystemClipboard();
    clipboard.setContents(selection, selection);
  }
}
