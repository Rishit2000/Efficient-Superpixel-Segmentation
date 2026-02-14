source "https://rubygems.org"

# GitHub Pages gem - this ensures compatibility with GitHub Pages
gem "github-pages", group: :jekyll_plugins

# If you want to use Jekyll natively (without GitHub Pages gem), use:
# gem "jekyll", "~> 4.3"

# Jekyll plugins (these are included in github-pages gem)
group :jekyll_plugins do
  gem "jekyll-feed"
  gem "jekyll-seo-tag"
end

# Windows and JRuby specific
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

# Performance-booster for watching directories on Windows
gem "wdm", "~> 0.1", :platforms => [:mingw, :x64_mingw, :mswin]

# Lock `http_parser.rb` gem to `v0.6.x` on JRuby builds
gem "http_parser.rb", "~> 0.6.0", :platforms => [:jruby]
