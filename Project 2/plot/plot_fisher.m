% plotting code for Fisher's Linear Discriminant

% the two 2x5 matrices
x1 = [4 1; 2 4; 2 3; 3 6; 4 4]'; 
x2 = [9 10; 6 8; 9 5; 8 7; 10 8]';

% FILL IN YOUR CODE FOR COMPUTING THE PROJECTION DIRECTION w HERE

% plot points
figure
hold on
scatter(x1(1, :), x1(2, :))
scatter(x2(1, :), x2(2, :), 'filled')

% plot discriminant direction as a vector
slope = w(2)/w(1);
p1 = [0 0];                         % First Point
p2 = [10 10*slope];                 % Second Point
dp = p2-p1;                         % Difference
quiver(p1(1), p1(2), dp(1), dp(2), 0)

% plot a grid over the points and vector
grid
axis([0 10 0 10])
text(p1(1), p1(2), sprintf('(%.0f,%.0f)', p1))
text(p2(1), p2(2), sprintf('(%.0f,%.0f)', p2))
