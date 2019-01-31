package pagerank;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;

public class RankComparator extends WritableComparator {
	
	public RankComparator() {
		super(Text.class, true);
	}	
	
	public int compare(WritableComparable o1, WritableComparable o2) {
		Text key1 = (Text) o1;
		Text key2 = (Text) o2;

        return key1.charAt(0)-key2.charAt(0);
	}
}
