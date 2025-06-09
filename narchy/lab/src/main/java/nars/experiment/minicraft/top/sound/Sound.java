package nars.experiment.minicraft.top.sound;

import jcog.exe.Exe;
import nars.experiment.minicraft.top.TopDownMinicraft;

import java.applet.Applet;
import java.applet.AudioClip;

public class Sound {
    public static final Sound playerHurt = new Sound("playerhurt.wav");
    public static final Sound playerDeath = new Sound("death.wav");
    public static final Sound monsterHurt = new Sound("monsterhurt.wav");
    public static final Sound test = new Sound("test.wav");
    public static final Sound pickup = new Sound("pickup.wav");
    public static final Sound bossdeath = new Sound("bossdeath.wav");
    public static final Sound craft = new Sound("craft.wav");

    private AudioClip clip;

    private Sound(String name) {
        try {
            clip = Applet.newAudioClip(TopDownMinicraft.class.getResource(name));
        } catch (RuntimeException e) {
            e.printStackTrace();
        }
    }

    public void play() {
//        try {
            //new Thread(() -> clip.play()).start();
            Exe.run(clip::play);
//        } catch (RuntimeException e) {
//            e.printStackTrace();
//        }
    }
}