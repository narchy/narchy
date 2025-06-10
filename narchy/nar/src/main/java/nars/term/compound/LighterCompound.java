//package nars.term.compound;
//
//import nars.Op;
//import nars.subterm.Subterms;
//import nars.subterm.TermList;
//import nars.term.Compound;
//import nars.term.Term;
//import nars.term.Unverified;
//
///** mutable, use with caution; hashCode is dynamically computed
// *
// * TODO partially functional
// * */
//public class LighterCompound extends TermList implements Unverified {
//    private final byte op;
//
//    private LighterCompound(byte op) {
//        super();
//        this.op = op;
//    }
//
//
//    public LighterCompound(Op op) {
//        this(op.id);
//    }
//
//    public LighterCompound(Op op, Term... subs) {
//        super(subs);
//        this.op = op.id;
//    }
//
//    @Override
//    public boolean equals(Object obj) {
//        if (this == obj) return true;
//        if (obj instanceof Compound) {
//            Compound c = (Compound)obj;
//            if (c.opID() == opID() && c.dt()==dt()) {
//                int s = c.subs();
//                if (subs()==s) {
//                    for (int i = 0; i < s; i++) {
//                        if (!sub(i).equals(c.sub(i)))
//                            return false;
//                    }
//                    return true;
//                }
//            }
//        }
//        return false;
//
//    }
//
//    @Override
//    public Term[] arrayShared() {
//        return arrayKeep();
//    }
//
//    @Override
//    public final int hashCode() {
//        return hashWith(opID());
//    }
//
//
//    @Override
//    public String toString() {
//        return Compound.toString(this);
//    }
//
//    @Override
//    public final Op op() {
//        return Op.the(op);
//    }
//
//    @Override
//    public final int opID() {
//        return op;
//    }
//
//
//    @Override
//    public Subterms subterms() {
//        return this;
//    }
//
//
//}
