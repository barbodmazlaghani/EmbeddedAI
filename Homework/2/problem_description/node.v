`timescale 1ns/1ns

module node(
    clk, in_ready, x, out, out_ready
);

parameter LAYERID=1;
parameter WIDTH = 17;
parameter FRACTION = 14;
parameter W_WIDTH = 9;
parameter INPUT_NUM = 196;
parameter OUTPUT_WIDTH = 25;
parameter NODEID=1;

parameter INPUT_LEN_DEPTH = $clog2(WIDTH * INPUT_NUM);
parameter INPUT_NUM_DEPTH = $clog2(INPUT_NUM);

input wire clk;
input wire in_ready;
input wire [WIDTH * INPUT_NUM - 1 : 0] x;

output reg signed [OUTPUT_WIDTH - 1 : 0] out;
output reg out_ready;

reg signed [WIDTH * (INPUT_NUM + 1) - 1 : 0] xx;


reg [INPUT_NUM_DEPTH : 0] i;
reg [INPUT_LEN_DEPTH : 0] j;
reg signed [OUTPUT_WIDTH - 1 : 0] sum;
reg signed [OUTPUT_WIDTH - 1 : 0] r;
reg signed [WIDTH - 1 : 0] signed_x;

wire [W_WIDTH * (INPUT_NUM + 1) -1 : 0] w_mem;
reg signed [W_WIDTH - 1 : 0] w;

weights_memory #(
		.INPUT_NUMBER(INPUT_NUM),
		.W_WIDTH(W_WIDTH),
		.LAYERID(LAYERID),
		.NODEID(NODEID)
) wm (
		.data(w_mem)
);

always @(posedge clk) begin
    if (in_ready) begin
      xx = {1 << FRACTION, x};
		sum = 0;
		if (INPUT_NUM == 1) begin
		    signed_x = x;
            w = w_mem[0 +: W_WIDTH];
			r = w * signed_x;
            w = w_mem[W_WIDTH +: W_WIDTH];
			sum = r + sum + w;
		end
		else
		    for (i = 0; i <= INPUT_NUM; i= i + 1) begin
				j = i * WIDTH;
				signed_x = xx[j+:WIDTH];
                w = w_mem[i * W_WIDTH +: W_WIDTH];
				r = w * signed_x;
				sum = r + sum;
		    end
        out <= sum;
        out_ready <= 1'b1;
    end else begin
        out_ready <= 1'b0;
    end
end

endmodule
