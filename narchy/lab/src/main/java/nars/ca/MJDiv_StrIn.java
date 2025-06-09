package nars.ca;




import java.util.StringTokenizer;
import java.util.Vector;

public class MJDiv_StrIn {

	private String m_Str;
	public boolean m_Active;
	public int m_X;
    public int m_Y;
	public boolean m_Repeat;
	public final Vector m_Vals = new Vector();
	public int m_Pos; 
						

	private void AddVal(Integer n) {
		if ((n >= 0) && (n <= MJBoard.MAX_CLO)) {
			m_Vals.addElement(n); 
		}
	}

	
	
	private void SetStr(String sStr) {
		int i;

		m_Str = sStr;

		m_Pos = 0;
        StringTokenizer st = new StringTokenizer(sStr, ",", false);
		while (st.hasMoreTokens()) {
			String sTok = st.nextToken();
			sTok = sTok.trim();
			if (sTok.contains("(")) 
			{
				int iPos = sTok.indexOf('(');
				String sBff = sTok.substring(0, iPos);
				int iCnt = Integer.parseInt(sBff);
				sBff = sTok.substring(iPos + 1); 
				iPos = sBff.indexOf(')');
				if (iPos >= 0)
					sBff = sBff.substring(0, iPos);
				while (iCnt > 0) {
					AddVal(Integer.valueOf(sBff));
					iCnt--;
				}
			} else 
			{
				AddVal(Integer.valueOf(sTok));
			}
		}
	}

	
	public void Reset() {
		m_Active = false;
		m_Repeat = true;
		m_X = 0;
		m_Y = 0;
		m_Str = "";
		m_Vals.removeAllElements();
	}

	
	
	public void SetFromString(String sStr) {

		Reset();

        StringTokenizer st = new StringTokenizer(sStr, ",", false);
		while (st.hasMoreTokens()) {
			String sTok = st.nextToken().toUpperCase();

			if (sTok.startsWith("ACT="))
				m_Active = Integer.parseInt(sTok.substring(4)) != 0;
			else if (sTok.startsWith("REP="))
				m_Repeat = Integer.parseInt(sTok.substring(4)) != 0;
			else if (sTok.startsWith("X="))
				m_X = Integer.parseInt(sTok.substring(2));
			else if (sTok.startsWith("Y="))
				m_Y = Integer.parseInt(sTok.substring(2));
		}

        int iPos = sStr.indexOf("str=");
		if (iPos >= 0) {
			String sBff = sStr.substring(iPos + 4);
			SetStr(sBff);
		}
	}

	
	public String GetAsString() {
		String sRet = "#STRIN";

		sRet = sRet + ",act=" + (m_Active ? "1" : "0");
		sRet = sRet + ",rep=" + (m_Repeat ? "1" : "0");
		sRet = sRet + ",x=" + m_X;
		sRet = sRet + ",y=" + m_Y;
		sRet = sRet + ",str=" + m_Str;

		return sRet;
	}
	

}
