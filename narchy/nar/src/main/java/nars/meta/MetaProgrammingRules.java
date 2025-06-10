package nars.meta;

public class MetaProgrammingRules {

    // --- TEMPORAL_INDUCTION_IMPL_BIDI ---

    /**
     * Rule to read the NAL.temporal.TEMPORAL_INDUCTION_IMPL_BIDI flag.
     * This rule, when processed as a question, will attempt to unify ?is_bidi
     * with the current boolean value of the flag.
     * Example Narsese input to trigger reading: `(<?is_bidi --> (^getTemporalInductionImplBidi)>)?`
     */
    public static final String READ_TEMPORAL_INDUCTION_IMPL_BIDI_QUERY = "(<?is_bidi --> (^getTemporalInductionImplBidi)>)?";

    /**
     * Rule to set the NAL.temporal.TEMPORAL_INDUCTION_IMPL_BIDI flag to False.
     * This operation is triggered by a belief that <Self --> set_bidi_false>.
     * The argument {False} is passed to the ^setTemporalInductionImplBidi functor.
     * Example Narsese input to trigger: `<Self --> set_bidi_false>.`
     */
    public static final String SET_TEMPORAL_INDUCTION_IMPL_BIDI_FALSE_RULE = "<<Self --> set_bidi_false> =/> <({False}) --> (^setTemporalInductionImplBidi)>>.";

    /**
     * Rule to set the NAL.temporal.TEMPORAL_INDUCTION_IMPL_BIDI flag to True.
     * This operation is triggered by a belief that <Self --> set_bidi_true>.
     * The argument {True} is passed to the ^setTemporalInductionImplBidi functor.
     * Example Narsese input to trigger: `<Self --> set_bidi_true>.`
     */
    public static final String SET_TEMPORAL_INDUCTION_IMPL_BIDI_TRUE_RULE = "<<Self --> set_bidi_true> =/> <({True}) --> (^setTemporalInductionImplBidi)>>.";


    // --- NOVEL_DURS ---

    /**
     * Rule to read the NAL.belief.NOVEL_DURS value.
     * This rule, when processed as a question, will attempt to unify ?novel_durs_value
     * with the current float value of the parameter.
     * Example Narsese input to trigger reading: `(<?novel_durs_value --> (^getNovelDurs)>)?`
     */
    public static final String READ_NOVEL_DURS_QUERY = "(<?novel_durs_value --> (^getNovelDurs)>)?";

    /**
     * Rule to set the NAL.belief.NOVEL_DURS value (e.g., to 2.5).
     * This operation is triggered by a belief that <Self --> set_novel_durs_custom>.
     * The argument {2.5} (as a float Term) is passed to the ^setNovelDurs functor.
     * Note: Narsese representation for floats might vary; typically handled as atomic terms.
     * We assume the parser and QuantityTerm can handle "2.5" or a similar representation.
     * Example Narsese input to trigger: `<Self --> set_novel_durs_custom>.`
     */
    public static final String SET_NOVEL_DURS_RULE = "<<Self --> set_novel_durs_custom> =/> <({2.5}) --> (^setNovelDurs)>>.";


    // --- COMPOUND_VOLUME_MAX ---

    /**
     * Rule to read the NAL.term.COMPOUND_VOLUME_MAX value.
     * This rule, when processed as a question, will attempt to unify ?volume_max_value
     * with the current integer value of the parameter.
     * Example Narsese input to trigger reading: `(<?volume_max_value --> (^getCompoundVolumeMax)>)?`
     */
    public static final String READ_COMPOUND_VOLUME_MAX_QUERY = "(<?volume_max_value --> (^getCompoundVolumeMax)>)?";

    /**
     * Rule to set the NAL.term.COMPOUND_VOLUME_MAX value (e.g., to 256).
     * This operation is triggered by a belief that <Self --> set_volume_max_custom>.
     * The argument {256} (as an Int Term) is passed to the ^setCompoundVolumeMax functor.
     * Example Narsese input to trigger: `<Self --> set_volume_max_custom>.`
     */
    public static final String SET_COMPOUND_VOLUME_MAX_RULE = "<<Self --> set_volume_max_custom> =/> <({256}) --> (^setCompoundVolumeMax)>>.";

}
