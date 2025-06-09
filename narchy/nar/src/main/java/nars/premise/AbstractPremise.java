package nars.premise;

public abstract class AbstractPremise extends Premise {

    public final int hash;

    protected AbstractPremise(int hash) {
        this.hash = hash;
    }

    @Override
    public final int hashCode() {
        return hash;
    }
}