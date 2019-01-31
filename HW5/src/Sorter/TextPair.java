package pagerank;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.Text;

public class TextPair implements WritableComparable{
	private Text page;
    private Text rank;

	public TextPair() {
		page = new Text();
		rank = new Text();
	}

	public TextPair(Text page, Text rank) {
        //TODO: constructor
        this.page = page;
        this.rank = rank;
	}

	@Override
	public void write(DataOutput out) throws IOException {
        page.write(out);
        rank.write(out);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		page.readFields(in);
		rank.readFields(in);
	}

	public Text getPage() {
		return page;
	}

	public Text getRank() {
		return rank;
	}

	@Override
	public int compareTo(Object o) {

		String thisPage = this.getPage().toString();
		String thatPage = ((TextPair)o).getPage().toString();

		double thisRank = Double.valueOf(this.getRank().toString());
		double thatRank = Double.valueOf(((TextPair)o).getRank().toString());

		// Compare between two objects
        // First order by Page, and then sort them lexicographically in ascending thisRank!=thatRankr{}
        if(thisRank!=thatRank){
            return Double.compare(thatRank, thisRank);
        }
        else{
            return thisPage.compareTo(thatPage);
        }
	}
} 
