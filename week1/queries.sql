-- Basic stats
select min(time), max(time), count(*), avg(score), max(score)
from hacker_news.items
where type='story' and dead!=false;

-- Basic score histogram
select score, count(*)
from hacker_news.items and dead!=false
where type='story'
group by score
order by score;

-- Score histogram by year
select extract(year from time) as year, score, count(*)
from hacker_news.items
where type='story' and dead!=false
group by extract(year from time), score
order by year, score;

-- Number of users by year
select extract(year from time) as year, count(distinct "by") as users from hacker_news.items
group by extract(year from time);

-- Number of users by month
select extract(month FROM time) as month, extract(year from time) as year, count(distinct "by") as users from hacker_news.items
group by extract(month FROM time), extract(year from time)
order by year, month;

-- Pearson correlation between hour and score
select corr((extract(hour from time)), score) from hacker_news.items where type='story' and dead!=false;

-- Pearson correlation between raw timestamp and score
select corr((extract(epoch from time)), score) from hacker_news.items where type='story' and dead!=false;




select sum(length(text)) from hacker_news.items where text is not null
and CARDINALITY(kids) > 2 and dead is null;

select text from hacker_news.items where length(text) > 2048
and CARDINALITY(kids) > 4 and dead is null;

select sum(length(title)) from hacker_news.items where title is not null and dead is null
and score > 2

select max(length(title)),  min(length(title)), avg(length(title))
from hacker_news.items where title is not null and dead is null and parent is null and type='story';

select id, "by", time, text, url, score, title
from hacker_news.items
where dead is null and type='story';

select percentile_cont(0.5) within group (order by score) from hacker_news.items where title is not null
and dead is null;

select count(distinct "by") from hacker_news.items
where "by" not in (select id from hacker_news.users);

WITH story_stats AS (
  SELECT
    "by" AS user_id,
    COUNT(*) AS story_count,
    MAX(score) AS max_score,
    MIN(score) AS min_score,
    AVG(score) AS mean_score,
    MAX(descendants) AS max_post_descendants,
    MIN(descendants) AS min_post_descendants,
    AVG(descendants) AS mean_post_descendants
  FROM hacker_news.items
  WHERE type = 'story' AND dead IS NULL AND title IS NOT NULL AND "by" IS NOT NULL
  GROUP BY "by"
),

comment_stats AS (
  SELECT
    "by" AS user_id,
    COUNT(*) AS comment_count,
    MAX(CARDINALITY(kids)) AS max_comment_children,
    MIN(CARDINALITY(kids)) AS min_comment_children,
    AVG(CARDINALITY(kids)) AS mean_comment_children,
    MAX(length(text)) AS max_comment_length,
    MIN(length(text)) AS min_comment_length,
    AVG(length(text)) AS mean_comment_length
  FROM hacker_news.items
  WHERE type = 'comment' AND dead IS NULL AND text IS NOT NULL AND "by" IS NOT NULL
  GROUP BY "by"
)

SELECT
  COALESCE(s.user_id, c.user_id, u.id) AS user_id,
  u.created,
  u.karma,
  CARDINALITY(u.submitted) AS length_submitted,

  s.story_count,
  s.max_score, s.min_score, s.mean_score,
  s.max_post_descendants, s.min_post_descendants, s.mean_post_descendants,

  COALESCE(c.comment_count, 0) AS comment_count,
  COALESCE(c.max_comment_children, 0) AS max_comment_children,
  COALESCE(c.min_comment_children, 0) AS min_comment_children,
  COALESCE(c.mean_comment_children, 0) AS mean_comment_children,
  COALESCE(c.max_comment_length, 0) AS max_comment_length,
  COALESCE(c.min_comment_length, 0) AS min_comment_length,
  COALESCE(c.mean_comment_length, 0)

FROM story_stats s
FULL OUTER JOIN comment_stats c ON s.user_id = c.user_id
FULL OUTER JOIN hacker_news.users u 
  ON COALESCE(s.user_id, c.user_id) = u.id
  LIMIT 10;

CREATE TABLE
  hacker_news.users (
    id character varying(255) NOT NULL,
    created timestamp without time zone NULL,
    karma integer NULL,
    about text NULL,
    submitted integer[] NULL
  );

CREATE TABLE
  hacker_news.items (
    id integer NOT NULL,
    dead boolean NULL,
    type
      character varying(20) NULL,
      by character varying(255) NULL,
      "time" timestamp without time zone NULL,
      text text NULL,
      parent integer NULL,
      kids integer[] NULL,
      url character varying(255) NULL,
      score integer NULL,
      title character varying(255) NULL,
      descendants integer NULL
  );