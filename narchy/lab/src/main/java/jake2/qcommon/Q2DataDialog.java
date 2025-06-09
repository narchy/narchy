/*
 * Q2DataDialog.java
 * Copyright (C)  2003
 */

package jake2.qcommon;

import jake2.Globals;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.net.URL;
import java.net.URLConnection;
import java.util.Enumeration;
import java.util.Locale;
import java.util.Vector;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;


class Q2DataDialog extends JDialog {

    private static final String home = System.getProperty("user.home");
    private static final String sep = System.getProperty("file.separator");

    Q2DataDialog() {
        super();
        initComponents();

        DisplayMode mode = GraphicsEnvironment.getLocalGraphicsEnvironment().getDefaultScreenDevice().getDisplayMode();
        int x = (mode.getWidth() - getWidth()) / 2;
        int y = (mode.getHeight() - getHeight()) / 2;
        setLocation(x, y);
        dir = String.join(sep, home, "Jake2", "baseq2");
        jTextField1.setText(dir);
    }

    private void initComponents() {
        JComponent.setDefaultLocale(Locale.US);

        choosePanel = new JPanel();
        statusPanel = new JPanel();
        status = new JLabel("initializing Jake2...");
        jTextField1 = new JTextField();
        JButton changeButton = new JButton();
        JButton cancelButton = new JButton();
        JButton exitButton = new JButton();
        JButton okButton = new JButton();

        setDefaultCloseOperation(WindowConstants.DO_NOTHING_ON_CLOSE);
        setTitle("Jake2 - Bytonic Software");

        setResizable(false);
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent evt) {
                formWindowClosing(evt);
            }
        });

        choosePanel.setLayout(new GridBagLayout());
        choosePanel.setMaximumSize(new Dimension(400, 100));
        choosePanel.setMinimumSize(new Dimension(400, 100));
        choosePanel.setPreferredSize(new Dimension(400, 100));


        GridBagConstraints gridBagConstraints = new GridBagConstraints();
        gridBagConstraints.gridx = 0;
        gridBagConstraints.gridy = 0;
        gridBagConstraints.gridwidth = 1;
        gridBagConstraints.insets = new Insets(5, 5, 5, 5);
        gridBagConstraints.weightx = 0;
        gridBagConstraints.anchor = GridBagConstraints.SOUTHWEST;
        choosePanel.add(new JLabel("baseq2 directory"), gridBagConstraints);

        gridBagConstraints.gridx = 1;
        gridBagConstraints.gridy = 0;
        gridBagConstraints.gridwidth = 2;
        gridBagConstraints.fill = GridBagConstraints.BOTH;
        gridBagConstraints.insets = new Insets(5, 2, 5, 2);
        gridBagConstraints.weightx = 1;
        choosePanel.add(jTextField1, gridBagConstraints);

        changeButton.setText("...");
        changeButton.addActionListener(this::changeButtonActionPerformed);
        gridBagConstraints.gridx = 3;
        gridBagConstraints.gridy = 0;
        gridBagConstraints.gridwidth = 1;
        gridBagConstraints.weightx = 0;
        gridBagConstraints.fill = GridBagConstraints.NONE;
        gridBagConstraints.insets = new Insets(5, 2, 5, 5);
        gridBagConstraints.anchor = GridBagConstraints.EAST;
        choosePanel.add(changeButton, gridBagConstraints);

        gridBagConstraints.gridx = 0;
        gridBagConstraints.gridy = 1;
        gridBagConstraints.gridwidth = 4;
        gridBagConstraints.weightx = 0;
        gridBagConstraints.weighty = 1;
        gridBagConstraints.fill = GridBagConstraints.VERTICAL;
        choosePanel.add(new JPanel(), gridBagConstraints);

        cancelButton.setText("Cancel");
        cancelButton.addActionListener(this::cancelButtonActionPerformed);
        gridBagConstraints.gridx = 0;
        gridBagConstraints.gridy = 2;
        gridBagConstraints.gridwidth = 4;
        gridBagConstraints.weighty = 0;
        gridBagConstraints.insets = new Insets(5, 5, 5, 5);
        gridBagConstraints.anchor = GridBagConstraints.SOUTH;
        choosePanel.add(cancelButton, gridBagConstraints);

        exitButton.setText("Exit");
        exitButton.addActionListener(Q2DataDialog::exitButtonActionPerformed);
        gridBagConstraints.gridx = 0;
        gridBagConstraints.gridy = 2;
        gridBagConstraints.gridwidth = 1;
        gridBagConstraints.anchor = GridBagConstraints.SOUTHWEST;
        choosePanel.add(exitButton, gridBagConstraints);

        okButton.setText("OK");
        okButton.addActionListener(this::okButtonActionPerformed);
        gridBagConstraints.gridx = 2;
        gridBagConstraints.gridy = 2;
        gridBagConstraints.gridwidth = 2;
        gridBagConstraints.anchor = GridBagConstraints.SOUTHEAST;
        choosePanel.add(okButton, gridBagConstraints);


        Jake2Canvas c = new Jake2Canvas();
        getContentPane().add(c, BorderLayout.CENTER);

        statusPanel.setLayout(new GridBagLayout());
        statusPanel.setMaximumSize(new Dimension(400, 100));
        statusPanel.setMinimumSize(new Dimension(400, 100));
        statusPanel.setPreferredSize(new Dimension(400, 100));
        gridBagConstraints = new GridBagConstraints();
        gridBagConstraints.gridx = 0;
        gridBagConstraints.gridy = 0;
        gridBagConstraints.gridwidth = 1;
        gridBagConstraints.fill = GridBagConstraints.HORIZONTAL;
        gridBagConstraints.insets = new Insets(10, 10, 10, 10);
        gridBagConstraints.weightx = 1.0;
        statusPanel.add(status, gridBagConstraints);
        getContentPane().add(statusPanel, BorderLayout.SOUTH);

        progressPanel = new ProgressPanel(this);
        installPanel = new InstallPanel(this);
        notFoundPanel = new NotFoundPanel(this);

        pack();
    }

    private void cancelButtonActionPerformed(ActionEvent evt) {
        showNotFoundPanel();
    }

    private static void exitButtonActionPerformed(ActionEvent evt) {
        System.exit(1);
    }

    private void okButtonActionPerformed(ActionEvent evt) {
        dir = jTextField1.getText();
        if (dir != null) {
            Cvar.Set("cddir", dir);
            FS.setCDDir();
        }

        synchronized (this) {
            notifyAll();
        }
    }

    private void changeButtonActionPerformed(ActionEvent evt) {
        JFileChooser chooser = new JFileChooser();
        chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        chooser.setDialogType(JFileChooser.CUSTOM_DIALOG);
        chooser.setMultiSelectionEnabled(false);
        chooser.setDialogTitle("choose a valid baseq2 directory");
        chooser.showDialog(this, "OK");

        dir = null;
        try {
            dir = chooser.getSelectedFile().getCanonicalPath();
        } catch (Exception e) {
        }
        if (dir != null) jTextField1.setText(dir);
        else dir = jTextField1.getText();

    }

    private static void formWindowClosing(WindowEvent evt) {

        System.exit(1);

        
    }

    private JPanel choosePanel;
    private JPanel statusPanel;
    private ProgressPanel progressPanel;
    private InstallPanel installPanel;
    private NotFoundPanel notFoundPanel;
    private JLabel status;
    private JTextField jTextField1;
    

    private String dir;

    private void showChooseDialog() {
        getContentPane().remove(statusPanel);
        getContentPane().remove(progressPanel);
        getContentPane().remove(installPanel);
        getContentPane().remove(notFoundPanel);
        getContentPane().add(choosePanel, BorderLayout.SOUTH);
        validate();
        repaint();
    }

    private void showStatus() {
        getContentPane().remove(choosePanel);
        getContentPane().remove(installPanel);
        getContentPane().add(statusPanel, BorderLayout.SOUTH);
        validate();
        repaint();
    }

    private void showProgressPanel() {
        getContentPane().remove(choosePanel);
        getContentPane().remove(installPanel);
        getContentPane().add(progressPanel, BorderLayout.SOUTH);
        validate();
        repaint();
    }

    private void showInstallPanel() {
        getContentPane().remove(choosePanel);
        getContentPane().remove(statusPanel);
        getContentPane().remove(notFoundPanel);
        getContentPane().add(installPanel, BorderLayout.SOUTH);
        validate();
        repaint();
    }

    private void showNotFoundPanel() {
        getContentPane().remove(choosePanel);
        getContentPane().remove(installPanel);
        getContentPane().remove(statusPanel);
        getContentPane().add(notFoundPanel, BorderLayout.SOUTH);
        validate();
        repaint();
    }

    void setStatus(String text) {
        status.setText(text);
    }

    void testQ2Data() {
        while (FS.LoadFile("pics/colormap.pcx") == null) {
            showNotFoundPanel();

            try {
                synchronized (this) {
                    wait();
                }
            } catch (InterruptedException e) {
            }
        }
        showStatus();
        repaint();
    }

    static class Jake2Canvas extends Canvas {
        private Image image;

        Jake2Canvas() {
            setSize(400, 200);
            try {
                image = ImageIO.read(getClass().getResource("/splash.png"));
            } catch (Exception e) {
            }

        }


        @Override
        public void paint(Graphics g) {
            g.drawImage(image, 0, 0, null);
        }

    }

    static class NotFoundPanel extends JPanel {

        private final Q2DataDialog parent;
        private JRadioButton dir;
        private JRadioButton install;
        private JButton ok;
        private JLabel message;

        NotFoundPanel(Q2DataDialog d) {
            parent = d;
            initComponents();
        }

        private void initComponents() {
            GridBagConstraints constraints = new GridBagConstraints();
            setLayout(new GridBagLayout());
            Dimension d = new Dimension(400, 100);
            setMinimumSize(d);
            setMaximumSize(d);
            setPreferredSize(d);

            message = new JLabel("Quake2 level data not found");
            message.setForeground(Color.RED);
            constraints.gridx = 0;
            constraints.gridy = 0;
            constraints.gridwidth = 2;
            constraints.insets = new Insets(5, 5, 2, 5);
            constraints.anchor = GridBagConstraints.CENTER;
            add(message, constraints);

            constraints.gridx = 1;
            constraints.gridy = 1;
            constraints.gridwidth = 2;
            constraints.weightx = 1;
            constraints.fill = GridBagConstraints.HORIZONTAL;
            constraints.insets = new Insets(0, 2, 0, 5);
            constraints.anchor = GridBagConstraints.WEST;
            JLabel label = new JLabel("select baseq2 directory from existing Quake2 installation");
            label.addMouseListener(new MouseAdapter() {
                @Override
                public void mouseClicked(MouseEvent e) {
                    dir.setSelected(true);
                }
            });
            add(label, constraints);

            constraints.gridx = 1;
            constraints.gridy = 2;
            label = new JLabel("download and install Quake2 demo data (38MB)");
            label.addMouseListener(new MouseAdapter() {
                @Override
                public void mouseClicked(MouseEvent e) {
                    install.setSelected(true);
                }
            });
            add(label, constraints);

            ButtonGroup selection = new ButtonGroup();
            dir = new JRadioButton();
            install = new JRadioButton();
            selection.add(dir);
            selection.add(install);

            constraints.gridx = 0;
            constraints.gridy = 1;
            constraints.gridwidth = 1;
            constraints.weightx = 0;
            constraints.insets = new Insets(0, 5, 0, 2);
            constraints.fill = GridBagConstraints.NONE;
            constraints.anchor = GridBagConstraints.EAST;
            dir.setSelected(true);
            add(dir, constraints);

            constraints.gridx = 0;
            constraints.gridy = 2;
            add(install, constraints);

            constraints.gridx = 0;
            constraints.gridy = 3;
            constraints.gridwidth = 2;
            constraints.weighty = 1;
            constraints.insets = new Insets(5, 5, 5, 5);
            constraints.fill = GridBagConstraints.NONE;
            constraints.anchor = GridBagConstraints.SOUTHWEST;
            JButton exit = new JButton("Exit");
            exit.addActionListener(e -> System.exit(0));









            }


        private void ok() {
            if (dir.isSelected()) {
                parent.showChooseDialog();
            } else {
                parent.showInstallPanel();
            }
        }
    }

    static class InstallPanel extends JPanel {

        private final Vector mirrorNames = new Vector();
        private final Vector mirrorLinks = new Vector();
        private final Q2DataDialog parent;
        private JComboBox mirrorBox;
        private JTextField destDir;

        InstallPanel(Q2DataDialog d) {
            initComponents();
            String dir = Q2DataDialog.home + Q2DataDialog.sep + "Jake2";
            destDir.setText(dir);
            initMirrors();
            parent = d;
        }

        private void initComponents() {
            GridBagConstraints constraints = new GridBagConstraints();
            setLayout(new GridBagLayout());
            Dimension d = new Dimension(400, 100);
            setMinimumSize(d);
            setMaximumSize(d);
            setPreferredSize(d);

            constraints.gridx = 0;
            constraints.gridy = 0;
            constraints.insets = new Insets(5, 5, 0, 5);
            constraints.anchor = GridBagConstraints.SOUTHWEST;
            add(new JLabel("download mirror"), constraints);

            constraints.gridx = 0;
            constraints.gridy = 1;
            constraints.insets = new Insets(5, 5, 5, 5);
            add(new JLabel("destination directory"), constraints);

            constraints.gridx = 1;
            constraints.gridy = 0;
            constraints.weightx = 1;
            constraints.gridwidth = 3;
            constraints.insets = new Insets(5, 5, 0, 5);
            constraints.fill = GridBagConstraints.HORIZONTAL;
            mirrorBox = new JComboBox();
            add(mirrorBox, constraints);

            constraints.gridx = 1;
            constraints.gridy = 1;
            constraints.gridwidth = 2;
            constraints.fill = GridBagConstraints.BOTH;
            constraints.insets = new Insets(5, 5, 5, 5);
            destDir = new JTextField();
            add(destDir, constraints);

            constraints.gridx = 3;
            constraints.gridy = 1;
            constraints.weightx = 0;
            constraints.gridwidth = 1;
            constraints.fill = GridBagConstraints.NONE;
            JButton choose = new JButton("...");
            choose.addActionListener(e -> choose());
            add(choose, constraints);

            constraints.gridx = 0;
            constraints.gridy = 2;
            constraints.gridwidth = 1;
            constraints.weighty = 1;
            constraints.fill = GridBagConstraints.NONE;
            JButton exit = new JButton("Exit");
            exit.addActionListener(e -> exit());
            add(exit, constraints);

            constraints.gridx = 0;
            constraints.gridy = 2;
            constraints.gridwidth = 4;
            constraints.anchor = GridBagConstraints.SOUTH;
            JButton cancel = new JButton("Cancel");
            cancel.addActionListener(e -> cancel());
            add(cancel, constraints);

            constraints.gridx = 2;
            constraints.gridy = 2;
            constraints.gridwidth = 2;
            constraints.anchor = GridBagConstraints.SOUTHEAST;
            JButton install = new JButton("Install");
            install.addActionListener(e -> install());
            add(install, constraints);
        }

        private void readMirrors() {
            InputStream in = getClass().getResourceAsStream("/mirrors");
            try (in; BufferedReader r = new BufferedReader(new InputStreamReader(in))) {
                while (true) {
                    String name = r.readLine();
                    String value = r.readLine();
                    if (name == null || value == null) break;
                    mirrorNames.add(name);
                    mirrorLinks.add(value);
                }
            } catch (Exception e) {
            }
        }

        private void initMirrors() {
            readMirrors();
            for (Object mirrorName : mirrorNames) {
                mirrorBox.addItem(mirrorName);
            }
            int i = Globals.rnd.nextInt(mirrorNames.size());
            mirrorBox.setSelectedIndex(i);
        }

        private void cancel() {
            parent.showNotFoundPanel();
        }

        private void install() {
            parent.progressPanel.destDir = destDir.getText();
            parent.progressPanel.mirror = (String) mirrorLinks.get(mirrorBox.getSelectedIndex());
            parent.showProgressPanel();
            new Thread(parent.progressPanel).start();
        }

        private static void exit() {
            
            System.exit(0);
            
        }

        private void choose() {
            JFileChooser chooser = new JFileChooser();
            chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
            chooser.setDialogType(JFileChooser.CUSTOM_DIALOG);
            chooser.setMultiSelectionEnabled(false);
            chooser.setDialogTitle("choose destination directory");
            chooser.showDialog(this, "OK");

            String dir = null;
            try {
                dir = chooser.getSelectedFile().getCanonicalPath();
            } catch (Exception e) {
            }
            if (dir != null) destDir.setText(dir);
        }
    }

    static class ProgressPanel extends JPanel implements Runnable {

        static final byte[] buf = new byte[8192];
        String destDir;
        String mirror;

        final JProgressBar progress = new JProgressBar();
        final JLabel label = new JLabel("");
        final JButton cancel = new JButton("Cancel");
        final Q2DataDialog parent;
        boolean running;

        ProgressPanel(Q2DataDialog d) {
            initComponents();
            parent = d;
        }

        void initComponents() {
            progress.setMinimum(0);
            progress.setMaximum(100);
            progress.setStringPainted(true);
            setLayout(new GridBagLayout());
            new GridBagConstraints();

            GridBagConstraints gridBagConstraints = new GridBagConstraints();
            gridBagConstraints.gridx = 0;
            gridBagConstraints.gridy = 0;
            gridBagConstraints.gridwidth = 1;
            gridBagConstraints.fill = GridBagConstraints.HORIZONTAL;
            gridBagConstraints.insets = new Insets(5, 10, 5, 10);
            gridBagConstraints.weightx = 1.0;
            gridBagConstraints.anchor = GridBagConstraints.SOUTH;
            add(label, gridBagConstraints);

            gridBagConstraints.gridy = 1;
            gridBagConstraints.anchor = GridBagConstraints.NORTH;
            add(progress, gridBagConstraints);

            gridBagConstraints.gridy = 1;
            gridBagConstraints.anchor = GridBagConstraints.SOUTH;
            gridBagConstraints.fill = GridBagConstraints.NONE;
            gridBagConstraints.weighty = 1;
            gridBagConstraints.weightx = 0;
            cancel.addActionListener(e -> cancel());
            add(cancel, gridBagConstraints);

            Dimension d = new Dimension(400, 100);
            setMinimumSize(d);
            setMaximumSize(d);
            setPreferredSize(d);
        }

        synchronized void cancel() {
            running = false;
        }

        @Override
        public void run() {
            synchronized (this) {
                running = true;
            }

            label.setText("downloading...");

            File dir = null;
            try {
                dir = new File(destDir);
                dir.mkdirs();
            } catch (Exception e) {
            }
            try {
                if (!dir.isDirectory() || !dir.canWrite()) {
                    endInstall("can't write to " + destDir);
                    return;
                }
            } catch (Exception e) {
                endInstall(e.getMessage());
                return;
            }

            File outFile;
            OutputStream out = null;
            InputStream in = null;
            try {
                URL url = new URL(mirror);
                URLConnection conn = url.openConnection();
                int length = conn.getContentLength();
                progress.setMaximum(length / 1024);
                progress.setMinimum(0);

                in = conn.getInputStream();

                outFile = File.createTempFile("Jake2Data", ".zip");
                outFile.deleteOnExit();
                out = new FileOutputStream(outFile);

                copyStream(in, out);
            } catch (Exception e) {
                endInstall(e.getMessage());
                return;
            } finally {
                try {
                    in.close();
                } catch (Exception e) {
                }
                try {
                    out.close();
                } catch (Exception e) {
                }
            }

            try {
                installData(outFile.getCanonicalPath());
            } catch (Exception e) {
                endInstall(e.getMessage());
                return;
            }


            try {
                if (outFile != null) outFile.delete();
            } catch (Exception e) {
            }

            endInstall("installation successful");
        }


        void installData(String filename) throws Exception {
            InputStream in = null;
            OutputStream out = null;
            try {
                ZipFile f = new ZipFile(filename);
                Enumeration e = f.entries();
                while (e.hasMoreElements()) {
                    ZipEntry entry = (ZipEntry) e.nextElement();
                    String name = entry.getName();
                    int i;
                    if ((i = name.indexOf("/baseq2")) > -1 && !name.contains(".dll")) {
                        name = destDir + name.substring(i);
                        File outFile = new File(name);
                        if (entry.isDirectory()) {
                            outFile.mkdirs();
                        } else {
                            label.setText("installing " + outFile.getName());
                            progress.setMaximum((int) entry.getSize() / 1024);
                            progress.setValue(0);
                            outFile.getParentFile().mkdirs();
                            out = new FileOutputStream(outFile);
                            in = f.getInputStream(entry);
                            copyStream(in, out);
                        }
                    }
                }
            } finally {
                try {
                    in.close();
                } catch (Exception e1) {
                }
                try {
                    out.close();
                } catch (Exception e1) {
                }
            }
        }

        void endInstall(String message) {
            parent.notFoundPanel.message.setText(message);
            parent.jTextField1.setText(destDir + "/baseq2");
            parent.showChooseDialog();
            parent.okButtonActionPerformed(null);
        }

        void copyStream(InputStream in, OutputStream out) throws Exception {
            try (in; out) {
                int c = 0;
                int l;
                while ((l = in.read(buf)) > 0) {
                    if (!running) throw new Exception("installation canceled");
                    out.write(buf, 0, l);
                    c += l;
                    int k = c / 1024;
                    progress.setValue(k);
                    progress.setString(k + "/" + progress.getMaximum() + " KB");
                }
            }
        }
    }

}