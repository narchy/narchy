/*
 * tuProlog - Copyright (C) 2001-2002  aliCE team at deis.unibo.it
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */
package alice.tuprolog.lib;

import alice.tuprolog.PrologError;
import alice.tuprolog.PrologLib;
import alice.tuprolog.Term;
import alice.tuprolog.Var;

/**
 * Library for managing DCGs.
 * 
 * Library/Theory dependency: BasicLibrary
 * 
 * 
 * 
 */
public class DCGLibrary extends PrologLib {


    @Override
    public String getTheory() {
        return """
                :- op(1200, xfx, '-->').\s
                :- op(200, xfx, '\\').\s
                dcg_nonterminal(X) :- list(X), !, fail.\s
                dcg_nonterminal(_).\s
                dcg_terminals(Xs) :- list(Xs).\s
                phrase(C,L) :- phrase_guard(C,L), phrase0(C,L).\s
                phrase(C,L,R) :- phrase_guard(C,L,R), phrase0(C,L,R).\s
                phrase0(Category, String, Left) :- dcg_parse(Category, String \\ Left).\s
                phrase0(Category, [H | T]) :- dcg_parse(Category, [H | T] \\ []).\s
                phrase0(Category,[]) :- dcg_parse(Category, [] \\ []).\s
                dcg_parse(A, Tokens) :- dcg_nonterminal(A), (A --> B), dcg_parse(B, Tokens).\s
                dcg_parse((A, B), Tokens \\ Xs) :- dcg_parse(A, Tokens \\ Tokens1), dcg_parse(B, Tokens1 \\ Xs).\s
                dcg_parse(A, Tokens) :- dcg_terminals(A), dcg_connect(A, Tokens).\s
                dcg_parse({A}, Xs \\ Xs) :- call(A).\s
                dcg_connect([], Xs \\ Xs).\s
                dcg_connect([W | Ws], [W | Xs] \\ Ys) :- dcg_connect(Ws, Xs \\ Ys).\s
                """;
    }

    

    public boolean phrase_guard_2(Term arg0, Term arg1) throws PrologError {
        arg0 = arg0.term();
        if (arg0 instanceof Var)
            throw PrologError.instantiation_error(prolog, 1);
        return true;
    }

    public boolean phrase_guard_3(Term arg0, Term arg1, Term arg2) throws PrologError {
        arg0 = arg0.term();
        if (arg0 instanceof Var)
            throw PrologError.instantiation_error(prolog, 1);
        return true;
    }

}