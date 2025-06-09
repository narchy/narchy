/*
 * Copyright 2012-present Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License. You may obtain
 * a copy of the License at
 *
 *     http:
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

package jcog.io;

import com.google.common.collect.FluentIterable;

import java.io.PrintStream;
import java.util.Collections;

/** Encapsulates the specifics of writing to a particular kind of terminal.
 *  https:
 * */
public final class Ansi {

  private static final String RESET = "\u001B[0m";

  private static final String BOLD = "\u001B[1m";

  private static final String BLACK = "\u001B[30m";
  private static final String GREY = "\u001B[0;37m";

  private static final String RED = "\u001B[31m";
  private static final String YELLOW = "\u001B[33m";
  private static final String GREEN = "\u001B[32m";
  private static final String CYAN = "\u001B[36m";

  private static final String ERROR_SEQUENCE = RED;
  private static final String WARNING_SEQUENCE = YELLOW;
  private static final String SUCCESS_SEQUENCE = GREEN;
  private static final String INFORMATION_SEQUENCE = CYAN;
  private static final String SUBTLE_SEQUENCE = GREY;

  private static final String BACKGROUND_YELLOW = "\u001B[43m";
  private static final String BACKGROUND_GREEN = "\u001B[42m";

  private static final String HIGHLIGHTED_ERROR_SEQUENCE = RED;
  private static final String HIGHLIGHTED_WARNING_SEQUENCE = BOLD + BACKGROUND_YELLOW + BLACK;
  private static final String HIGHLIGHTED_SUCCESS_SEQUENCE = BOLD + BACKGROUND_GREEN + BLACK;

  private static final String CURSOR_PREVIOUS_LINE = "\u001B[%dA";

  private static final String ERASE_IN_LINE = "\u001B[%dK";

  private static final String STOP_WRAPPING = "\u001B[?7l";
  private static final String RESUME_WRAPPING = "\u001B[?7h";

  private static final int ANSI_PREVIOUS_LINE_STRING_CACHE_MAX_LINES = 22;
  private static final String[] ANSI_PREVIOUS_LINE_STRING_CACHE =
      new String[ANSI_PREVIOUS_LINE_STRING_CACHE_MAX_LINES];

  private static final String ANSI_ERASE_LINE = String.format(ERASE_IN_LINE, 2);
  private static final String ANSI_ERASE_TO_THE_END_OF_LINE = String.format(ERASE_IN_LINE, 0);

  private final boolean isAnsiTerminal;

  private final String clearLineString;

  private static final Ansi noTtyAnsi = new Ansi(false /* isAnsiTerminal */);
  private static final Ansi forceTtyAnsi = new Ansi(true /* isAnsiTerminal */);

  /** Construct an Ansi object and conditionally enable fancy escape sequences. */
  public Ansi(boolean isAnsiTerminal) {
    this.isAnsiTerminal = isAnsiTerminal;
    clearLineString = isAnsiTerminal ? ANSI_ERASE_LINE : "";
  }

  public static Ansi withoutTty() {
    return noTtyAnsi;
  }

  public static Ansi forceTty() {
    return forceTtyAnsi;
  }

  public boolean isAnsiTerminal() {
    return isAnsiTerminal;
  }

  public String asErrorText(String text) {
    return wrapWithColor(ERROR_SEQUENCE, text);
  }

  public String asWarningText(String text) {
    return wrapWithColor(WARNING_SEQUENCE, text);
  }

  public String asSuccessText(String text) {
    return wrapWithColor(SUCCESS_SEQUENCE, text);
  }

  public String asSubtleText(String text) {
    return wrapWithColor(SUBTLE_SEQUENCE, text);
  }

  public String asInformationText(String text) {
    return wrapWithColor(INFORMATION_SEQUENCE, text);
  }

  public String asGreenText(String text) {
    return wrapWithColor(GREEN, text);
  }

  public String asRedText(String text) {
    return wrapWithColor(RED, text);
  }

  public String getHighlightedWarningSequence() {
    return isAnsiTerminal ? HIGHLIGHTED_ERROR_SEQUENCE : "";
  }

  public String getHighlightedResetSequence() {
    return isAnsiTerminal ? RESET : "";
  }

  public String asHighlightedFailureText(String text) {
    return wrapWithColor(HIGHLIGHTED_ERROR_SEQUENCE, text);
  }

  public String asHighlightedWarningText(String text) {
    return wrapWithColor(HIGHLIGHTED_WARNING_SEQUENCE, text);
  }

  public String asHighlightedSuccessText(String text) {
    return wrapWithColor(HIGHLIGHTED_SUCCESS_SEQUENCE, text);
  }

  public Iterable<String> asNoWrap(Iterable<String> textParts) {
	  return isAnsiTerminal ? FluentIterable.from(Collections.singleton(STOP_WRAPPING))
		  .append(textParts)
		  .append(RESUME_WRAPPING) : textParts;
  }

  public void printHighlightedSuccessText(PrintStream stream, String text) {
    stream.print(asHighlightedSuccessText(text));
  }

  public void printlnHighlightedFailureText(PrintStream stream, String text) {
    stream.println(asHighlightedFailureText(text));
  }

  public String asHighlightedStatusText(SeverityLevel level, String text) {
      return switch (level) {
          case OK -> asHighlightedSuccessText(text);
          case WARNING -> asHighlightedWarningText(text);
          case ERROR -> asHighlightedFailureText(text);
      };
  }

  /** Moves the cursor {@code y} lines up. */
  public String cursorPreviousLine(int y) {
    if (!isAnsiTerminal) {
      return "";
    }

    if (y >= ANSI_PREVIOUS_LINE_STRING_CACHE_MAX_LINES) {
      return String.format(CURSOR_PREVIOUS_LINE, y);
    }

    synchronized (ANSI_PREVIOUS_LINE_STRING_CACHE) {
      if (ANSI_PREVIOUS_LINE_STRING_CACHE[y] == null) {
        ANSI_PREVIOUS_LINE_STRING_CACHE[y] = String.format(CURSOR_PREVIOUS_LINE, y);
      }

      return ANSI_PREVIOUS_LINE_STRING_CACHE[y];
    }
  }

  /** Clears the line the cursor is currently on. */
  public String clearLine() {
    return clearLineString;
  }

  /** Clears from the cursor to the end of line. */
  public static String clearToTheEndOfLine() {
    return ANSI_ERASE_TO_THE_END_OF_LINE;
  }

  public enum SeverityLevel {
    OK,
    WARNING,
    ERROR
  }

  private String wrapWithColor(String color, String text) {
    if (!isAnsiTerminal || text.isEmpty()) {
      return text;
    }

    
    if (text.charAt(0) != '\t') {
      return color + text + RESET;
    }

    
    int firstNonTab = indexOfFirstNonTab(text);
    return firstNonTab == -1 ?
            color + text + RESET :
            text.substring(0, firstNonTab) + color + text.substring(firstNonTab) + RESET;
  }

  /** @return the index of the first character that's not a tab, or -1 if none is found. */
  private static int indexOfFirstNonTab(String s) {
      int n = s.length();
      for (int i = 1; i < n; i++)
          if (s.charAt(i) != '\t') return i;
      return -1;
  }

    /**
     * Created by me on 9/12/15.
     */
    public enum ANSI {
        ;

        public static final String COLOR_RESET = "\u001b[0m";

        public static final String RED = "\u001b[1;31m";
        public static final String COLOR_WARNING = "\u001b[1;33m";
        public static final String COLOR_INFO = "\u001b[0;32m";
        public static final String COLOR_CONFIG = "\u001b[1;34m";
        public static final String COLOR_FINE = "\u001b[0;36m";
        public static final String COLOR_FINER = "\u001b[0;35m";
        public static final String COLOR_FINEST = "\u001b[1;30m";
    }
}