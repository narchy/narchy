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
public class EDCGLibrary extends PrologLib {

    @Override
    public String getTheory() {
        return """
                :- op(1200, xfx, '==>').
                :- op(200, xfx, '\\').\s
                :- op(200, xfx, ';').\s
                :- op(200, fx, '*'). %Zero or more productions\s
                :- op(200, fx, '+'). %One or more productions\s
                :- op(200, fx, '?'). %Zero or one production\s
                :- op(200, fx, '^'). %Exactly N productions (for parsing only)\s
                :- op(200, fx, '#'). %Exactly N productions (for AST generation)\s
                edcg_parse(*(A,_,[]),LO \\ LO).\s
                edcg_parse(*(A,X,[X|L]), LI \\ LO) :- edcg_parse(A, LI \\ L1), edcg_parse(*(A,X,L),L1 \\ LO).\s
                edcg_parse(*(A), LI \\ LO) :- ((edcg_parse(A,LI \\ L1), LI\\=L1, edcg_parse(*A, L1 \\ LO));LI=LO).\s
                edcg_parse(+(A,X,[X|L]), LI \\ LO) :- edcg_parse(A, LI \\ L1), edcg_parse(*(A,X,L),L1 \\ LO).\s
                edcg_parse(+(A), LI \\ LO) :- (edcg_parse(A,LI \\ L1), LI\\=L1, edcg_parse(*A,L1 \\ LO)).\s
                edcg_parse(?(A,_,E2,E2), LO \\ LO).\s
                edcg_parse(?(A,E1,_,E1), LI \\ LO) :- edcg_parse(A, LI \\ LO).\s
                edcg_parse(?(A),LI \\ LO) :- edcg_parse(A, LI \\ LO);LI=LO.\s
                edcg_parse((A;B), Tokens) :- edcg_parse(A, Tokens);edcg_parse(B, Tokens).\s
                edcg_parse(#(A,N,X,L), LI \\ LO) :- edcg_power(#(A,N,0,X,L),LI \\ LO).\s
                edcg_power(#(A,N,N,_,[]),LO \\ LO).\s
                edcg_power(#(A,N,M,X,[X|L]), LI \\ LO) :- M1 is M+1, !,edcg_parse(A, LI \\ L1), edcg_power(#(A,N,M1,X,L),L1 \\ LO).\s
                edcg_parse(^(A,N), LI \\ LO) :- edcg_power(^(A,N,0),LI \\ LO).\s
                edcg_power(^(A,N,N),LO \\ LO).\s
                edcg_power(^(A,N,M), LI \\ LO) :- M1 is M+1, !,edcg_parse(A, LI \\ L1), edcg_power(^(A,N,M1),L1 \\ LO).\s
                edcg_nonterminal(X) :- list(X), !, fail.\s
                edcg_nonterminal(_).\s
                edcg_terminals(Xs) :- list(Xs).\s
                edcg_phrase(Category, String, Left) :- edcg_parse(Category, String \\ Left).\s
                edcg_phrase(Category, [H | T]) :- edcg_parse(Category, [H | T] \\ []).\s
                edcg_phrase(Category,[]) :- edcg_parse(Category, [] \\ []).\s
                edcg_parse(A, Tokens) :- edcg_nonterminal(A), (A ==> B), edcg_parse(B, Tokens).\s
                edcg_parse((A, B), Tokens \\ Xs) :- edcg_parse(A, Tokens \\ Tokens1), edcg_parse(B, Tokens1 \\ Xs).\s
                edcg_parse(A, Tokens) :- edcg_terminals(A), edcg_connect(A, Tokens).\s
                edcg_parse({A}, Xs \\ Xs) :- call(A).\s
                edcg_connect([], Xs \\ Xs).\s
                edcg_connect([W | Ws], [W | Xs] \\ Ys) :- edcg_connect(Ws, Xs \\ Ys).\s
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