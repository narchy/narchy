/*
 * Created on Nov 5, 2003
 * 
 * Copyright (C)aliCE team at deis.unibo.it
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
 *
 */
package alice.tuprolog;

public class TestLibrary extends StringLibrary {

	@Override
	public String getName() {
		return "TestLibraryName";
	}

	public Term sum_2(NumberTerm arg0, NumberTerm arg1){
		return new NumberTerm.Int(arg0.intValue() + arg1.intValue());
	}
	
	public boolean println_1(Term arg0){
		prolog.output(arg0.toString());
		return true;
	}
	
}
