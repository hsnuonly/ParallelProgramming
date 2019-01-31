package pagerank;

import java.util.ArrayList;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.fs.FileSystem;


public class PageRank{
	public long N;
    public double pr_dangling;
    public int partition;
    
    public PageRank(int n){
        this.partition = n;
    }
    public static void main(String[] args) throws Exception {  
    
        String input = "/user/ta/PageRank/input-"+args[0];
        String output = "pr_"+args[0];
        int iter = Integer.valueOf(args[1]);
        if(iter==-1)iter=Integer.MAX_VALUE;

        PageRank pageRank = new PageRank(8);
        WikiParser parser = new WikiParser(pageRank);
        WikiParser2 parser2 = new WikiParser2(pageRank);
        Ranker rank = new Ranker(pageRank);
        Sorter sorter = new Sorter(pageRank);

        ArrayList<Double> errors = new ArrayList<>();
        int i = 0;
        parser.run(input,output);
        parser2.run(output);
        System.out.println(pageRank.N*pageRank.pr_dangling);

        double error = Double.MAX_VALUE;

        while(true){
            boolean conv = true;
            for(i=0;i<iter;i++){
                if(error<0.001)break;
                error = rank.run(output,i);
                System.out.println(error);
                if(errors.size()>0&&error>errors.get(errors.size()-1)*1.1){
                    conv = false;
                    break;
                }
                if(errors.size()>i)errors.set(i,error);
                else errors.add(error);
            }
            if(conv)break;
        }

        sorter.run(output,i);
        for(Double e:errors)
            System.out.println(e);
        System.exit(0);
    }

    public void setN(long N){
        this.N = N;
    }

    public void setPR_dangling(double pr_dangling){
        this.pr_dangling = pr_dangling;
    }
}
